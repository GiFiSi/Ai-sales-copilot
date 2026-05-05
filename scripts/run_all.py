"""Script CLI — Sobe API + Frontend em paralelo.

Inicia o servidor FastAPI (backend) e o Streamlit (frontend)
em processos separados.

Uso:
    python scripts/run_all.py
    python scripts/run_all.py --api-port 8000 --st-port 8501
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

# Adiciona raiz do projeto ao path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Inicia API e frontend em paralelo."""
    parser = argparse.ArgumentParser(
        description="Inicia o AI Sales Copilot (API + Frontend)",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Porta do FastAPI (padrão: 8000)",
    )
    parser.add_argument(
        "--st-port",
        type=int,
        default=8501,
        help="Porta do Streamlit (padrão: 8501)",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Inicia apenas a API (sem frontend)",
    )
    parser.add_argument(
        "--frontend-only",
        action="store_true",
        help="Inicia apenas o frontend (sem API)",
    )

    args = parser.parse_args()

    processes = []

    logger.info("=" * 60)
    logger.info("🤖 AI Sales Copilot — Iniciando...")
    logger.info("=" * 60)

    try:
        # Inicia API
        if not args.frontend_only:
            api_cmd = [
                sys.executable, "-m", "uvicorn",
                "src.api.main:app",
                "--host", "0.0.0.0",
                "--port", str(args.api_port),
                "--reload",
            ]
            logger.info("🚀 API: http://localhost:%d", args.api_port)
            logger.info("📖 Docs: http://localhost:%d/docs", args.api_port)

            api_process = subprocess.Popen(
                api_cmd,
                cwd=str(PROJECT_ROOT),
                env={**os.environ},
            )
            processes.append(("API", api_process))

        # Inicia Frontend
        if not args.api_only:
            frontend_cmd = [
                sys.executable, "-m", "streamlit",
                "run", str(PROJECT_ROOT / "frontend" / "app.py"),
                "--server.port", str(args.st_port),
                "--server.headless", "true",
            ]
            logger.info("💬 Frontend: http://localhost:%d", args.st_port)

            frontend_process = subprocess.Popen(
                frontend_cmd,
                cwd=str(PROJECT_ROOT),
                env={**os.environ},
            )
            processes.append(("Frontend", frontend_process))

        logger.info("=" * 60)
        logger.info("Pressione Ctrl+C para encerrar.")
        logger.info("=" * 60)

        # Aguarda processos
        for name, proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        logger.info("\n⛔ Encerrando processos...")
        for name, proc in processes:
            logger.info("  Parando %s (PID %d)...", name, proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        logger.info("Todos os processos encerrados.")

    except Exception as e:
        logger.error("Erro: %s", e)
        for _, proc in processes:
            proc.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
