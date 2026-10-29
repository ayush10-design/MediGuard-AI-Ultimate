"""Evidence-limited packaging review; NEVER authenticates the physical medicine."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageOps

import config

MISSING = {'', 'not detected', 'not found', 'n/a', 'none'}


def field_value(fields, key):
    raw = (fields or {}).get(key) or {}
    value = str(raw.get('value', '')).strip()
    return '' if value.lower() in MISSING else value


def _hash_image(img):
    px = ImageOps.grayscale(img).resize((8, 8)).getdata()
    vals = list(px)
    mean = sum(vals) / len(vals)
    return sum((int(v >= mean) << (63 - i)) for i, v in enumerate(vals))


def check_local(img, root=None):
    """Match images only; dataset contains no authoritative product identity register."""
    root = Path(root or os.getenv('MEDIGUARD_REFERENCE_DATASET', str(Path.home() / 'Downloads' / 'dataset_clean'))).expanduser()
    if not root.exists():
        return {'status': 'unavailable', 'note': f'Local reference dataset not found: {root}', 'nearest': []}
    try:
        target = _hash_image(img)
        exact = hashlib.sha256(_jpeg_bytes(img)).hexdigest()  # diagnostic only; encoding may differ
        matches = []
        count = 0
        for split in ('train', 'val', 'test'):
            for label in ('Fake', 'Real'):
                folder = root / split / label
                if not folder.is_dir():
                    continue
                for path in folder.iterdir():
                    if path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.webp') or not path.is_file():
                        continue
                    try:
                        with Image.open(path) as ref:
                            d = (target ^ _hash_image(ref)).bit_count()
                        count += 1
                        matches.append((d, label, path.name, split))
                    except (OSError, ValueError):
                        continue
        if not matches:
            return {'status': 'unavailable', 'note': 'No readable local reference images.', 'nearest': []}
        matches.sort(key=lambda x: x[0])
        nearest = [{'distance': d, 'dataset_label': label, 'filename': name, 'split': split}
                   for d, label, name, split in matches[:3]]
        del exact
        return {'status': 'compared', 'count': count, 'nearest': nearest,
                'note': 'Perceptual image-hash similarity only. Nearby images can have different products or labels; '
                        'this is NOT a batch check or proof of authenticity.'}
    except Exception as exc:
        return {'status': 'unavailable', 'note': f'Dataset comparison failed: {type(exc).__name__}', 'nearest': []}


def _jpeg_bytes(img):
    from io import BytesIO
    buff = BytesIO()
    img.convert('RGB').save(buff, format='JPEG', quality=85)
    return buff.getvalue()


def check_online(fields, model='gemini-3.5-flash'):
    """Search public medicine information using Tavily."""
    import os
    import json
    import urllib.request
    import urllib.error
    from dotenv import load_dotenv

    load_dotenv(config.BASE_DIR / ".env")

    name = field_value(fields, "medicine_name")
    maker = field_value(fields, "manufacturer")
    strength = field_value(fields, "strength")

    if not name:
        return {
            "status": "unavailable",
            "summary": "Medicine name missing.",
            "sources": []
        }

    key = os.getenv("TAVILY_API_KEY", "").strip()

    if not key:
        return {
            "status": "unavailable",
            "summary": "Tavily API key not configured.",
            "sources": []
        }

    composition = (
        field_value(fields, "composition")
        or field_value(fields, "generic_composition")
        or ""
    )

    query = (
        f'"{name}" {strength} {composition} '
        "medicine tablet composition manufacturer India"
    )

    payload = json.dumps({
        "query": query[:400],
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": False,
        "include_raw_content": False
    }).encode("utf-8")

    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.load(response)

        import re

        def normalize(value):
            return re.sub(r"[^a-z0-9]", "", str(value).lower())

        def result_mentions_name(text):
            # Match brand and formulation separately from strength.
            name_tokens = re.findall(r"[a-z]+", str(name).lower())
            text_normalized = normalize(text)
            return bool(
                name_tokens
                and all(token in text_normalized for token in name_tokens)
            )

        results = []
        for item in data.get("results", []):
            url = str(item.get("url", ""))
            title = str(item.get("title", ""))
            content = str(item.get("content", ""))

            if not url.startswith("https://"):
                continue

            if result_mentions_name(title + " " + content):
                results.append({
                    "url": url,
                    "title": title[:160],
                    "content": content[:1500]
                })

        if not results:
            return {
                "status": "unverified",
                "summary": "No medicine-specific sources found.",
                "sources": [],
                "comparison": {}
            }

        # Retrieve full text from up to 3 relevant product pages.
        # Basic Tavily Extract uses free-tier API credits.
        try:
            extract_payload = json.dumps({
                "urls": [item["url"] for item in results[:3]],
                "extract_depth": "basic",
                "include_images": False
            }).encode("utf-8")

            extract_request = urllib.request.Request(
                "https://api.tavily.com/extract",
                data=extract_payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )

            with urllib.request.urlopen(
                extract_request, timeout=25
            ) as response:
                extracted = json.load(response)

            page_text = {
                item.get("url"): str(item.get("raw_content") or "")
                for item in extracted.get("results", [])
            }

            for item in results:
                if item["url"] in page_text:
                    item["content"] += " " + page_text[item["url"]][:12000]

        except Exception:
            # Search snippets remain available if extraction fails.
            pass

        evidence = " ".join(
            item["title"] + " " + item["content"]
            for item in results
        ).lower()

        comparison = {
            "medicine_name": "Found in public product information",
            "strength": "Not confirmed",
            "composition": "Not confirmed",
            "manufacturer": "Not confirmed"
        }

        # Match strength with units; accept 2mg, 2 mg and 2.0 MG.
        strength_values = re.findall(
            r"(\d+(?:\.\d+)?)\s*(mg|mcg|ug|iu)\b",
            str(strength).lower()
        )

        evidence_strengths = {
            (float(number), unit.lower())
            for number, unit in re.findall(
                r"(?<![\d.])(\d+(?:\.\d+)?)\s*(mg|mcg|ug|iu)\b",
                evidence,
                flags=re.IGNORECASE
            )
        }

        if strength_values and all(
            (float(number), unit) in evidence_strengths
            for number, unit in strength_values
        ):
            comparison["strength"] = "Found in public product information"
        elif strength_values:
            # Handle product titles such as "Vomikind MD 2mg".
            for item in results:
                title = item["title"].lower()
                title_values = {
                    (float(n), u.lower())
                    for n, u in re.findall(
                        r"(\d+(?:\.\d+)?)\s*(mg|mcg|ug|iu)\b",
                        title
                    )
                }
                if all(
                    (float(n), u) in title_values
                    for n, u in strength_values
                ):
                    comparison["strength"] = "Found in public product information"
                    break

        # Source-specific strength evidence.
        # Never match strength using unrelated medicine pages.
        def source_has_strength(item):
            text = (item["title"] + " " + item["content"]).lower()
            matches = re.findall(
                r"(?<![\d.])(\d+(?:\.\d+)?)\s*(mg|mcg|ug|iu)\b",
                text
            )
            available = {(float(n), u) for n, u in matches}
            return bool(strength_values) and all(
                (float(n), u) in available
                for n, u in strength_values
            )

        if any(source_has_strength(item) for item in results):
            comparison["strength"] = "Found in public product information"

        composition = (
            field_value(fields, "composition")
            or field_value(fields, "generic_composition")
            or field_value(fields, "generic_name")
            or ""
        )

        composition_lower = str(composition).lower()
        components = []

        if "calcium" in composition_lower:
            components.append("calcium")

        if "cholecalciferol" in composition_lower or "vitamin d3" in composition_lower:
            components.append("vitamin d3")

        # Additional commonly encountered generic medicine names.
        for ingredient in (
            "ondansetron",
            "paracetamol",
            "ibuprofen",
            "amoxicillin",
            "azithromycin",
            "metformin",
            "cetirizine"
        ):
            if ingredient in composition_lower:
                components.append(ingredient)

        if components:
            found = []
            if "calcium" in components and "calcium" in evidence:
                found.append("calcium")
            if "vitamin d3" in components and (
                "vitamin d3" in evidence
                or "cholecalciferol" in evidence
            ):
                found.append("vitamin d3")

            for ingredient in components:
                if ingredient not in ("calcium", "vitamin d3"):
                    if ingredient in evidence:
                        found.append(ingredient)

            if len(set(found)) == len(set(components)):
                comparison["composition"] = "Found in public product information"

        if maker and normalize(maker) in normalize(evidence):
            comparison["manufacturer"] = "Found in public product information"

        sources = [
            {"url": item["url"], "title": item["title"]}
            for item in results[:5]
        ]

        summary = (
            "Online product-information comparison (public search and extracted pages): "
            + " | ".join(
                f"{key.replace('_', ' ').title()}: {value}"
                for key, value in comparison.items()
            )
            + ". Search results do not verify the physical pack, "
              "batch number, expiry date or authenticity. "
              "Brand owner and contract manufacturer may differ."
        )

        return {
            "status": "sources_found",
            "summary": summary[:1200],
            "sources": sources,
            "comparison": comparison
        }

    except urllib.error.HTTPError as exc:
        return {
            "status": "unavailable",
            "summary": f"Tavily API HTTP {exc.code}.",
            "sources": []
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "summary": f"Online search failed: {type(exc).__name__}.",
            "sources": []
        }


def combined_assessment(fields, ml, local, online):
    """Screening decision deliberately never equates softmax with authenticity."""
    issues = []
    for key in ('medicine_name', 'manufacturer', 'batch_number', 'expiry_date'):
        if not field_value(fields, key):
            issues.append(f'Missing packaging field: {key.replace("_", " ")}')
    from utils.report_generator import parse_month_year
    from datetime import date
    exp = parse_month_year(field_value(fields, 'expiry_date'))
    mfg = parse_month_year(field_value(fields, 'mfg_date'))
    if exp and exp < date.today().replace(day=1):
        issues.append('Printed expiry appears to be in the past (confirm directly with pharmacist).')
    if mfg and exp and mfg > exp:
        issues.append('Printed manufacturing date appears later than expiry date.')
    if issues:
        outcome = 'Review packaging details'
    else:
        outcome = 'Needs Verification'
    return {
        'outcome': outcome,
        'reason': ('Potential packaging-data issue detected; independent verification required.' if issues else
                   'No trustworthy check of this physical pack or its batch has been performed.'),
        'flags': issues,
        'ml': {'label': ml.get('predicted_class') if ml.get('ok') else 'Not available',
               'scores': ml.get('scores', {}) if ml.get('ok') else {}},
        'local': local,
        'online': online,
        'disclaimer': 'This is educational screening. No image match, web listing or AI score proves a medicine genuine or counterfeit.'
    }


def combined_pdf(report, assessment):
    """A standalone combined screening PDF, independent of legacy report exporter."""
    from io import BytesIO
    from textwrap import wrap
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    output = BytesIO()
    doc = canvas.Canvas(output, pagesize=A4)
    _, height = A4
    y = height - 45
    def line(value, strong=False):
        nonlocal y
        doc.setFont('Helvetica-Bold' if strong else 'Helvetica', 10)
        for segment in wrap(str(value).encode('ascii', errors='replace').decode('ascii'), width=96) or ['']:
            if y < 55:
                doc.showPage(); y = height - 45
            doc.drawString(40, y, segment)
            y -= 15
    line('MediGuard AI - Combined Medicine Screening Report', True)
    line(f"Report ID: {report.get('id', '')} | Created: {report.get('created_at', '')}")
    line(f"Assessment: {assessment['outcome']}", True)
    line(assessment['reason'])
    line('Extracted packaging fields', True)
    for key, item in (report.get('fields') or {}).items():
        line(f"{item.get('label', key)}: {item.get('value', '')}")
    line('Visual model output (NOT proof of authenticity)', True)
    line(f"Class: {assessment['ml']['label']} | Scores: {assessment['ml']['scores']}")
    line('Local image similarity', True)
    line(assessment['local'].get('note', 'Unavailable'))
    for ref in assessment['local'].get('nearest', []):
        line(f"Nearest: {ref['dataset_label']} / distance={ref['distance']} / {ref['filename']}")
    line('Online product-information lookup', True)
    line(assessment['online'].get('summary', 'Not checked'))
    for ref in assessment['online'].get('sources', []):
        line(ref['url'])
    for flag in assessment.get('flags', []):
        line('Check: ' + flag)
    line(assessment['disclaimer'], True)
    doc.save()
    return output.getvalue()
