"""
Create a test license on the VectorBridge dashboard API.
Run this BEFORE the benchmark if you have the dashboard running.

Usage:
  # Start dashboard first:
  #   uvicorn dashboard.server:app --host 0.0.0.0 --port 8080

  python create_test_license.py [--api http://localhost:8080]
"""

import argparse
import json
import urllib.request
import urllib.parse


def create_license(api_base, org, plan, email):
    payload = json.dumps({"org": org, "plan": plan, "email": email}).encode()
    req = urllib.request.Request(
        f"{api_base}/v1/license/create",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="http://localhost:8080")
    p.add_argument("--org", default="VectorBridge Benchmark")
    p.add_argument("--plan", default="enterprise")
    p.add_argument("--email", default="insightits.info@gmail.com")
    args = p.parse_args()

    print(f"Creating {args.plan} license for {args.org} ...")
    result = create_license(args.api, args.org, args.plan, args.email)

    print("\n" + "═" * 50)
    print(f"  License Key: {result['license_key']}")
    print(f"  Plan:        {result['plan']}")
    print(f"  Org:         {result['org']}")
    print("═" * 50)
    print("\nUse this key in the benchmark:")
    print(f"  python run_benchmark.py --license-key {result['license_key']} ...")
    print(f"\nOr set env var:")
    print(f"  export LICENSE_KEY={result['license_key']}")

    # Save to file for convenience
    with open(".license_key", "w") as f:
        f.write(result["license_key"])
    print(f"\nKey also saved to .license_key")


if __name__ == "__main__":
    main()
