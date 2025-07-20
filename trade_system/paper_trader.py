# -*- coding: utf-8 -*-
"""Módulo de paper trading simples"""
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List
import logging
import random
import requests

logger = logging.getLogger(__name__)


def fetch_market_price(symbol: str) -> float:
    """Obtém preço atual do símbolo usando API pública da Binance"""
    url = "https://api.binance.com/api/v3/ticker/price"
    try:
        resp = requests.get(url, params={"symbol": symbol}, timeout=5)
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception as exc:
        logger.warning(f"Falha ao obter preço de {symbol}: {exc}")
        return 0.0


@dataclass
class PaperTrader:
    capital_inicial: float
    capital: float = field(init=False)
    posicoes: Dict[str, float] = field(default_factory=dict)
    historico: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.capital = self.capital_inicial
        logger.info(f"PaperTrader iniciado com capital inicial de ${self.capital_inicial:,.2f}")

    def obter_preco(self, symbol: str) -> float:
        return fetch_market_price(symbol)

    def comprar(self, symbol: str, quantidade: float):
        preco = self.obter_preco(symbol)
        custo = preco * quantidade
        if custo > self.capital:
            raise ValueError("Capital insuficiente para comprar")
        self.capital -= custo
        self.posicoes[symbol] = self.posicoes.get(symbol, 0.0) + quantidade
        self.historico.append({
            "tipo": "BUY", "symbol": symbol, "qtd": quantidade,
            "preco": preco, "timestamp": datetime.utcnow().isoformat()
        })
        logger.info(f"BUY {quantidade} {symbol} @ ${preco:,.2f} (capital: ${self.capital:,.2f})")

    def vender(self, symbol: str, quantidade: float):
        if self.posicoes.get(symbol, 0.0) < quantidade:
            raise ValueError("Posição insuficiente para vender")
        preco = self.obter_preco(symbol)
        self.capital += preco * quantidade
        self.posicoes[symbol] -= quantidade
        self.historico.append({
            "tipo": "SELL", "symbol": symbol, "qtd": quantidade,
            "preco": preco, "timestamp": datetime.utcnow().isoformat()
        })
        logger.info(f"SELL {quantidade} {symbol} @ ${preco:,.2f} (capital: ${self.capital:,.2f})")

    def resumo(self) -> Dict[str, Any]:
        valor_posicoes = sum(self.obter_preco(s) * q for s, q in self.posicoes.items())
        lucro = self.capital + valor_posicoes - self.capital_inicial
        return {
            "capital_restante": self.capital,
            "posicoes": {s: q for s, q in self.posicoes.items() if q},
            "lucro_prejuizo": lucro,
            "operacoes": len(self.historico)
        }

    def salvar_historico(self, caminho: str):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(self.historico, f, ensure_ascii=False, indent=2)
        logger.info(f"Histórico salvo em {caminho}")


def gerar_sinais_simulados(
    n: int = 5, symbol: str = "BTCUSDT", quantidade: float = 0.001
) -> Generator[Dict[str, Any], None, None]:
    """Gera sinais de compra/venda aleatórios para demonstração"""
    for _ in range(n):
        yield {
            "tipo": random.choice(["BUY", "SELL"]),
            "symbol": symbol,
            "quantidade": quantidade,
        }
