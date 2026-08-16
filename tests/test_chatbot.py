# tests/test_chatbot.py
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chatbot import process_query
from inventory_optimizer import load_engineered_data


def test_chatbot_empty_query():
    df = load_engineered_data()
    res = process_query("", df)
    assert "Please enter a question" in res


def test_chatbot_restock_query():
    df = load_engineered_data()
    res = process_query("Which products need restock urgently?", df)
    assert ("RESTOCK" in res or "SYSTEMS OPTIMAL" in res)


def test_chatbot_product_query():
    df = load_engineered_data()
    res = process_query("Tell me about iPhone 15 Pro", df)
    assert "iPhone 15 Pro" in res or "P101" in res


def test_chatbot_top_demand_query():
    df = load_engineered_data()
    res = process_query("What are the top demand leaders?", df)
    assert "TOP DEMAND VELOCITY LEADERS" in res


def test_chatbot_bundles_query():
    df = load_engineered_data()
    res = process_query("Recommend product bundles", df)
    assert "RECOMMENDED PRODUCT BUNDLES" in res
