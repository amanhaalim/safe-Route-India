#!/usr/bin/env python3
# =============================================================================
# quick_test.py — Test the routing engine from the command line
# =============================================================================
# Run after completing the pipeline:
#   python quick_test.py
#   python quick_test.py --city mumbai --hour 22
# =============================================================================

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_route_table(routes: dict):
    """Pretty-print a routes dict."""
    headers = ["Route", "Distance", "Safety%", "Crime", "Accident", "Flood", "Risk", "Tier"]
    rows = []
    for name, r in routes.items():
        if r:
            s = r["summary"]
            rows.append([
                name.upper(),
                f"{s['distance_km']:.2f} km",
                f"{s['safety_pct']:.1f}%",
                f"{s['avg_crime_score']:.3f}",
                f"{s['avg_accident_score']:.3f}",
                f"{s['avg_flood_score']:.3f}",
                f"{s['avg_risk_score']:.3f}",
                s["risk_label"],
            ])
        else:
            rows.append([name.upper(), "—", "—", "—", "—", "—", "—", "NO PATH"])

    col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt   = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)

    print("\n" + "─" * (sum(col_w) + len(col_w)*2 + 2))
    print(fmt.format(*headers))
    print("─" * (sum(col_w) + len(col_w)*2 + 2))
    for row in rows:
        print(fmt.format(*row))
    print("─" * (sum(col_w) + len(col_w)*2 + 2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick routing test")
    parser.add_argument("--city",    "-c", default="chennai")
    parser.add_argument("--origin",  "-o", default="Chennai Central Railway Station")
    parser.add_argument("--dest",    "-d", default="T. Nagar")
    parser.add_argument("--hour",    "-H", default=12,  type=int)
    parser.add_argument("--profile", "-p", default="default")
    args = parser.parse_args()

    print(f"\n🛡️  SafeRoute India — Quick Test")
    print(f"   City:        {args.city}")
    print(f"   Origin:      {args.origin}")
    print(f"   Destination: {args.dest}")
    print(f"   Hour:        {args.hour:02d}:00")
    print(f"   Profile:     {args.profile}")

    from src.routing import find_safe_routes
    try:
        routes, orig, dest = find_safe_routes(
            origin_address=args.origin,
            destination_address=args.dest,
            city_key=args.city,
            travel_hour=args.hour,
            profile=args.profile,
        )

        print(f"\n   Origin coords:      {orig}")
        print(f"   Destination coords: {dest}")
        print_route_table(routes)

        # Sanity check
        safest  = routes.get("safest")
        fastest = routes.get("fastest")
        if safest and fastest:
            risk_diff = fastest["summary"]["avg_risk_score"] - safest["summary"]["avg_risk_score"]
            dist_diff = safest["summary"]["distance_km"]     - fastest["summary"]["distance_km"]
            print(f"\n   Risk reduced by safest route: {risk_diff:+.3f}")
            print(f"   Extra distance for safety:   {dist_diff:+.2f} km")

    except FileNotFoundError as e:
        print(f"\n❌ Graph not ready: {e}")
        print("   Run the pipeline first:")
        print("   python run_pipeline.py --city " + args.city)
    except ValueError as e:
        print(f"\n❌ Geocoding failed: {e}")
        print("   Try a more specific address or check your internet connection.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
