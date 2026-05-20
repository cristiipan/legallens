"""Download the CUAD dataset.

CUAD (Contract Understanding Atticus Dataset) is the standard open-source
benchmark for contract clause extraction:
  - 510 contracts
  - 13,000+ expert annotations
  - 41 clause categories
  - Released under CC BY 4.0

Source: https://github.com/TheAtticusProject/cuad/
"""
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress

console = Console()

CUAD_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
DATA_DIR = Path(__file__).parent.parent / "data" / "cuad"


async def download() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = DATA_DIR / "data.zip"

    if output.exists():
        console.print(f"[yellow]CUAD data already downloaded at {output}[/yellow]")
        return

    console.print(f"Downloading CUAD from {CUAD_URL}...")
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        async with client.stream("GET", CUAD_URL) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with Progress() as progress:
                task = progress.add_task("[cyan]Downloading...", total=total)
                with output.open("wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

    console.print(f"[green]✓ Downloaded to {output}[/green]")
    console.print("[blue]Next: unzip and run `python scripts/ingest.py`[/blue]")


if __name__ == "__main__":
    import asyncio
    asyncio.run(download())
