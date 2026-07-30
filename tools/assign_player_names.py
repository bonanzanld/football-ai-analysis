from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from football_ai.tracking.entity_corrections import EntityRole, TeamAssignment
from football_ai.tracking.entity_identity import load_entity_identities
from football_ai.tracking.entity_roster import (
    PlayerProfile,
    TeamRoster,
    load_team_roster,
    save_team_roster,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Koppel namen aan spelers van je eigen team.")
    parser.add_argument("--video", required=True, help="Bestandsnaam in videos/ of volledig pad.")
    parser.add_argument("--own-team", choices=("team_a", "team_b"))
    parser.add_argument("--team-name", help="Naam van je eigen club of team.")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / "videos" / video_path
    entity_dir = PROJECT_ROOT / "output" / "entities"
    identities_path = entity_dir / f"{video_path.stem}_entity_identities.json"
    roster_path = entity_dir / f"{video_path.stem}_team_roster.json"
    if not identities_path.exists():
        raise FileNotFoundError(
            f"Speleridentiteiten ontbreken: {identities_path}. "
            "Maak eerst de fysieke speleridentiteiten."
        )
    identities = load_entity_identities(identities_path)
    existing = load_team_roster(roster_path) if roster_path.exists() else None

    print("\nSPELERSNAMEN EIGEN TEAM")
    print("Namen worden gekoppeld aan fysieke spelers, niet aan tijdelijke boxnummers.")
    print("De tegenstander blijft anoniem. Druk Enter om een bestaande naam te behouden.")
    team = _choose_team(args.own_team, existing)
    team_name = args.team_name or (existing.own_team_name if existing else "")
    while not team_name.strip():
        team_name = input("Naam van je eigen team: ").strip()

    previous = {item.identity_id: item for item in existing.players} if existing else {}
    candidates = [
        item for item in identities.identities
        if item.team is team and item.role in (EntityRole.PLAYER, EntityRole.GOALKEEPER)
    ]
    players = []
    for identity in candidates:
        old = previous.get(identity.identity_id)
        role = "keeper" if identity.role is EntityRole.GOALKEEPER else "speler"
        prompt = f"{identity.label} ({role}) - naam"
        if old:
            prompt += f" [{old.display_name}]"
        name = input(prompt + ": ").strip()
        if not name and old:
            name = old.display_name
        if not name:
            print("  Overgeslagen; het bestaande spelerslabel blijft zichtbaar.")
            continue
        number_prompt = "  Rugnummer"
        if old and old.squad_number:
            number_prompt += f" [{old.squad_number}]"
        number = input(number_prompt + " (optioneel): ").strip()
        if not number and old:
            number = old.squad_number
        players.append(
            PlayerProfile(
                identity_id=identity.identity_id,
                display_name=name,
                squad_number=number,
                position_periods=old.position_periods if old else (),
            )
        )

    roster = TeamRoster(
        source_video=identities.source_video,
        own_team_name=team_name.strip(),
        own_team=team,
        players=tuple(players),
    )
    save_team_roster(roster, roster_path)
    print(f"\nSpelerslijst opgeslagen: {roster_path}")
    print(f"Namen gekoppeld: {len(players)}/{len(candidates)} spelers van {team_name.strip()}")
    print("Draai daarna de balbezitanalyse opnieuw; namen verschijnen ook in de statistieken.")


def _choose_team(value: str | None, existing: TeamRoster | None) -> TeamAssignment:
    if value:
        return TeamAssignment(value)
    if existing:
        return existing.own_team
    while True:
        answer = input("Is je eigen team Team A of Team B? [A/B]: ").strip().lower()
        if answer in {"a", "team_a"}:
            return TeamAssignment.TEAM_A
        if answer in {"b", "team_b"}:
            return TeamAssignment.TEAM_B
        print("Vul A of B in. Kijk naar de bestaande videolabels om dit te bepalen.")


if __name__ == "__main__":
    main()
