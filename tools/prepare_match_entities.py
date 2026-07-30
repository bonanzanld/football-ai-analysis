from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_tool(tool: str, video: str, extra: tuple[str, ...] = ()) -> None:
    command = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        str(PROJECT_ROOT / "tools" / tool),
        "--video",
        video,
        *extra,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bereid alle wedstrijdentiteiten voor een nieuwe video in een vaste "
            "volgorde voor: detectie, personencontrole en keepercontrole."
        )
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--team-a-name", default="Team A (BLAUW kader)")
    parser.add_argument("--team-b-name", default="Team B (ROOD kader)")
    args = parser.parse_args()

    common_names = (
        "--team-a-name",
        args.team_a_name,
        "--team-b-name",
        args.team_b_name,
    )
    print("STAP 1/5 - Personen analyseren", flush=True)
    run_tool("analyze_entities.py", args.video, ("--seconds", str(args.seconds)))
    print("STAP 2/5 - Spelers, keepers en uitsluitingen controleren", flush=True)
    run_tool("review_entities.py", args.video, common_names)
    print("STAP 3/5 - Keeperkandidaten bepalen", flush=True)
    run_tool("analyze_goalkeepers.py", args.video)
    print("STAP 4/5 - Keeperkandidaten controleren", flush=True)
    run_tool("review_goalkeepers.py", args.video, common_names)
    print("STAP 5/5 - Definitieve entiteiten opnieuw opbouwen", flush=True)
    run_tool("analyze_entities.py", args.video, ("--seconds", str(args.seconds)))
    print("Voorbereiding voltooid. Deze video heeft nu eigen controles en uitsluitingen.")


if __name__ == "__main__":
    main()
