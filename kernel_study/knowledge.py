from .core import ROOT, read_json


def select_cards(family, measurements=None, excluded_category=None):
    """Stable IDs and order; identical selector for both knowledge arms.

    Family is always available evidence. No unavailable counters are inferred.
    Resource-pressure cards rank earlier when actual spill reports are nonzero.
    """
    bank = read_json(ROOT / 'knowledge/cards.json')
    cards = [c for c in bank['cards'] if ('all' in c['families'] or family in c['families'])
             and c['category'] != excluded_category]
    spills = any((x.get('spills') or 0) > 0 for x in (measurements or {}).get('kernels', []))
    return sorted(cards, key=lambda c: (0 if spills and c['category'] == 'resources' else 1, c['id']))
