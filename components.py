"""Reusable visual components for the TAKAL Streamlit dashboard."""

from __future__ import annotations

import base64
import html
import os

import plotly.graph_objects as go
import streamlit as st

from utils import BRAND_LOGO_PATH, COLORS


def apply_styles() -> None:
    brand_logo_b64 = ""
    if os.path.exists(BRAND_LOGO_PATH):
        with open(BRAND_LOGO_PATH, "rb") as brand_logo_file:
            brand_logo_b64 = base64.b64encode(brand_logo_file.read()).decode("ascii")

    st.markdown(
        f"""
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

          :root {{
            --takal-bg: #F2F5F3;
            --takal-white: #FFFFFF;
            --takal-border: #E3E9E5;
            --takal-text: #0B221B;
            --takal-muted: #6C827A;
            --takal-mint: #25C998;
            --takal-mint-light: #E7F8F2;
            --takal-mint-text: #0D7A58;
            --takal-forest-dark: #07231A;
            --takal-forest-pill: #0D2E24;
            --takal-forest-border: #1B473A;
            --takal-forest-card: #0E2A22;
          }}

          /* Global Typography */
          html, body, [class*="css"], .stApp {{
            color: var(--takal-text);
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
          }}

          .stApp {{
            background-color: var(--takal-bg);
          }}

          .stApp *:not([data-testid="stIconMaterial"]):not([class*="material"]):not(code):not(pre):not(kbd):not(svg):not(svg *) {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
          }}

          /* Keep Streamlit's icon glyphs on their icon font */
          [data-testid="stIconMaterial"],
          [data-testid="stSidebar"] [data-testid="stIconMaterial"],
          [data-testid="stSidebar"] span[class*="material-symbols"],
          span[class*="material-symbols"],
          span[class*="material-icons"] {{
            font-family: 'Material Symbols Rounded', 'Material Icons' !important;
          }}

          [data-testid="stAppViewContainer"] .main .block-container {{
            max-width: 1400px;
            padding: 2.2rem 2.8rem 3.5rem 2.8rem;
          }}

          /* --- SIDEBAR CONTAINER --- */
          [data-testid="stSidebar"] {{
            background-color: var(--takal-forest-dark) !important;
            border-right: none !important;
          }}

          [data-testid="stSidebar"] * {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
          }}

          [data-testid="stSidebar"],
          [data-testid="stSidebar"] > div:first-child,
          [data-testid="stSidebar"] [data-testid="stSidebarContent"],
          [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
            overflow: hidden !important;
            scrollbar-width: none !important;
          }}

          [data-testid="stSidebar"] *::-webkit-scrollbar {{
            display: none !important;
            width: 0 !important;
            height: 0 !important;
          }}

          [data-testid="stSidebar"] > div:first-child,
          [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
            height: 100dvh !important;
            max-height: 100dvh !important;
            min-height: 0 !important;
            padding: 0 !important;
            display: flex !important;
            flex-direction: column !important;
            box-sizing: border-box !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
            flex: 1 1 auto !important;
            min-height: 0 !important;
            padding: 0.4rem 1.4rem 1.8rem 1.4rem !important;
            display: flex !important;
            flex-direction: column !important;
            box-sizing: border-box !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div,
          [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
          [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div,
          [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
            flex: 1 1 auto !important;
            display: flex !important;
            flex-direction: column !important;
            min-height: 0 !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{
            position: absolute !important;
            top: 0.3rem !important;
            right: 0.4rem !important;
            height: auto !important;
            min-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            background: transparent !important;
            z-index: 5;
          }}

          /* Sidebar Logo Button */
          .st-key-home_reset {{
            display: flex !important;
            justify-content: center !important;
            margin: 0 0 1.4rem 0 !important;
          }}
          .st-key-home_reset button {{
            min-height: 9rem !important;
            width: 100% !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
          }}
          .st-key-home_reset button:hover,
          .st-key-home_reset button:focus,
          .st-key-home_reset button:active {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            transform: none !important;
          }}
          .st-key-home_reset button::before {{
            content: "";
            display: block;
            width: 100%;
            height: 9rem;
            margin: 0 auto;
            background: center center / contain no-repeat url("data:image/png;base64,{brand_logo_b64}");
            transition: transform 250ms cubic-bezier(0.4, 0, 0.2, 1);
          }}
          .st-key-home_reset button:hover::before {{
            transform: scale(1.03);
          }}
          .st-key-home_reset button p {{
            display: none !important;
          }}

          /* --- SIDEBAR DROPDOWN SPECIFICATION --- */
          [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
          [data-testid="stSidebar"] [data-testid="stSelectbox"] label p {{
            color: #7B9B8E !important;
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.1em !important;
            text-transform: uppercase !important;
            margin-bottom: 0.6rem !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {{
            background: transparent !important;
            border: none !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] div,
          [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
          [data-testid="stSidebar"] [data-testid="stSelectbox"] div[role="combobox"] {{
            background-color: var(--takal-forest-pill) !important;
            background: var(--takal-forest-pill) !important;
            color: #FFFFFF !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child {{
            border: 1px solid var(--takal-forest-border) !important;
            border-radius: 20px !important;
            min-height: 48px !important;
            height: 48px !important;
            padding: 0 16px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            box-sizing: border-box !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child:hover {{
            background-color: #11362B !important;
            border-color: #276250 !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
          [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
          [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
          [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within * {{
            background-color: var(--takal-forest-pill) !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child {{
            overflow: hidden !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div:first-child {{
            border-color: var(--takal-mint) !important;
            box-shadow: 0 0 0 3px rgba(37, 201, 152, 0.18) !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child::before {{
            content: "" !important;
            display: inline-block !important;
            width: 10px !important;
            height: 10px !important;
            min-width: 10px !important;
            border-radius: 50% !important;
            background-color: #25C998 !important;
            box-shadow: 0 0 0 4px rgba(37, 201, 152, 0.22) !important;
            margin-right: 12px !important;
            flex-shrink: 0 !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] span,
          [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] p,
          [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] [aria-selected="true"] {{
            color: #FFFFFF !important;
            font-size: 0.94rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            user-select: none !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] input {{
            caret-color: transparent !important;
            pointer-events: none !important;
            cursor: pointer !important;
            user-select: none !important;
            color: transparent !important;
            width: 0 !important;
            min-width: 0 !important;
            padding: 0 !important;
            border: none !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] svg {{
            fill: #7B9B8E !important;
            width: 1.15rem !important;
            height: 1.15rem !important;
            margin-left: auto !important;
          }}

          /* --- SIDEBAR COMBOBOX (React Aria) --- */
          [data-testid="stSidebar"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"] {{
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
            height: 48px !important;
            min-height: 48px !important;
            padding: 0 10px 0 16px !important;
            box-sizing: border-box !important;
            background: var(--takal-forest-pill) !important;
            border: 1px solid var(--takal-forest-border) !important;
            border-radius: 20px !important;
            box-shadow: none !important;
            cursor: pointer !important;
            transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"]:hover {{
            background: #11362B !important;
            border-color: #276250 !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"]:focus-within {{
            border-color: var(--takal-mint) !important;
            box-shadow: 0 0 0 3px rgba(37, 201, 152, 0.18) !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"]::before {{
            content: "" !important;
            display: block !important;
            width: 10px !important;
            height: 10px !important;
            min-width: 10px !important;
            flex: 0 0 10px !important;
            border-radius: 50% !important;
            background-color: #25C998 !important;
            box-shadow: 0 0 0 4px rgba(37, 201, 152, 0.22) !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] .react-aria-ComboBox input {{
            flex: 1 1 auto !important;
            min-width: 0 !important;
            height: 100% !important;
            padding: 0 !important;
            background: transparent !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            caret-color: transparent !important;
            font-size: 0.94rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em !important;
            cursor: pointer !important;
            user-select: none !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] .react-aria-ComboBox button {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            margin-left: auto !important;
            color: #7B9B8E !important;
          }}

          [data-testid="stSidebar"] [data-testid="stSelectbox"] .react-aria-ComboBox button svg {{
            fill: #7B9B8E !important;
            width: 1.15rem !important;
            height: 1.15rem !important;
          }}

          [data-testid="stSelectboxVirtualDropdown"] {{
            background: #0D2E24 !important;
            border: 1px solid #1B473A !important;
            border-radius: 16px !important;
            padding: 6px !important;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.55) !important;
          }}
          [data-testid="stSelectboxVirtualDropdown"] [role="listbox"] {{
            background: transparent !important;
          }}
          [data-testid="stSelectboxVirtualDropdown"] [role="option"] {{
            background: transparent !important;
            color: #B4CDC3 !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            border-radius: 10px !important;
            cursor: pointer !important;
          }}
          [data-testid="stSelectboxVirtualDropdown"] [role="option"] * {{
            color: inherit !important;
          }}
          [data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
          [data-testid="stSelectboxVirtualDropdown"] [role="option"][data-focused="true"],
          [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] {{
            background: #174537 !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
          }}

          div[data-baseweb="popover"] {{
            background: transparent !important;
          }}
          div[data-baseweb="popover"] [role="listbox"],
          div[data-baseweb="popover"] ul {{
            background-color: #0D2E24 !important;
            background: #0D2E24 !important;
            border: 1px solid #1B473A !important;
            border-radius: 16px !important;
            padding: 6px !important;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.55) !important;
          }}
          div[data-baseweb="popover"] [role="option"],
          div[data-baseweb="popover"] li {{
            background-color: transparent !important;
            color: #B4CDC3 !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            cursor: pointer !important;
          }}
          div[data-baseweb="popover"] [role="option"]:hover,
          div[data-baseweb="popover"] [role="option"][aria-selected="true"],
          div[data-baseweb="popover"] li:hover,
          div[data-baseweb="popover"] li[aria-selected="true"] {{
            background-color: #174537 !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
          }}

          [data-testid="stSidebar"] hr {{
            display: none !important;
          }}

          [data-testid="stSidebar"] .st-key-technical_details {{
            margin-top: auto !important;
            padding-top: 1.5rem !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
          }}

          [data-testid="stSidebar"] .st-key-technical_details [data-testid="stCheckbox"],
          [data-testid="stSidebar"] .st-key-technical_details [data-testid="stToggle"] {{
            width: auto !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
          }}

          [data-testid="stSidebar"] .st-key-technical_details label {{
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 0.75rem !important;
            width: auto !important;
            cursor: pointer !important;
            background: transparent !important;
          }}

          [data-testid="stSidebar"] .st-key-technical_details label p {{
            color: #FFFFFF !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            margin: 0 !important;
            line-height: 1.2 !important;
            background: transparent !important;
          }}

          [data-testid="stSidebar"] .st-key-technical_details [data-testid="stWidgetLabel"],
          [data-testid="stSidebar"] .st-key-technical_details [data-testid="stWidgetLabel"] * {{
            background: transparent !important;
            margin: 0 !important;
          }}

          [data-testid="stSidebar"] .st-key-technical_details label > div:not([data-testid="stWidgetLabel"]) {{
            flex: 0 0 auto !important;
            background-color: rgba(255, 255, 255, 0.16) !important;
            border-color: #7FA394 !important;
            box-shadow: inset 0 0 0 1.5px #7FA394 !important;
            transition: background-color 200ms ease, box-shadow 200ms ease !important;
          }}
          [data-testid="stSidebar"] .st-key-technical_details label:hover > div:not([data-testid="stWidgetLabel"]) {{
            background-color: rgba(255, 255, 255, 0.24) !important;
          }}
          [data-testid="stSidebar"] .st-key-technical_details label > div:not([data-testid="stWidgetLabel"]) > div {{
            background-color: #E8F5EF !important;
          }}

          [data-testid="stSidebar"] .st-key-technical_details label:has(input:checked) > div:not([data-testid="stWidgetLabel"]) {{
            background-color: var(--takal-mint) !important;
            border-color: var(--takal-mint) !important;
            box-shadow: none !important;
          }}

          [data-testid="stToggle"] input:checked + div {{
            background-color: var(--takal-mint) !important;
            border-color: var(--takal-mint) !important;
          }}

          /* --- PILL NAVIGATION TABS --- */
          .stTabs {{
            margin-bottom: 2rem;
          }}

          .stTabs [data-baseweb="tab-list"] {{
            background-color: #E6EBE8;
            padding: 5px;
            border-radius: 16px;
            gap: 4px;
            border-bottom: none !important;
            width: fit-content;
          }}

          .stTabs [data-baseweb="tab-highlight"] {{
            display: none !important;
          }}

          .stTabs [data-baseweb="tab"] {{
            border-radius: 12px !important;
            padding: 8px 20px !important;
            background: transparent !important;
            border: none !important;
            color: #556E65 !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
          }}

          .stTabs [data-baseweb="tab"]:hover {{
            color: var(--takal-text) !important;
          }}

          .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background-color: #FFFFFF !important;
            color: var(--takal-text) !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
          }}

          /* --- HEADINGS --- */
          .page-heading {{
            font-size: 2.35rem;
            font-weight: 700;
            letter-spacing: -0.035em;
            color: var(--takal-text);
            margin: 0 0 0.5rem 0;
            line-height: 1.15;
          }}

          .page-intro {{
            font-size: 0.95rem;
            color: var(--takal-muted);
            line-height: 1.55;
            margin-bottom: 2rem;
            max-width: 1100px;
          }}

          .page-intro strong {{
            color: var(--takal-text);
            font-weight: 600;
          }}

          .section-subheading {{
            font-size: 1.6rem;
            font-weight: 600;
            letter-spacing: -0.025em;
            color: var(--takal-text);
            margin: 2.2rem 0 0.3rem 0;
          }}

          .section-caption {{
            font-size: 0.92rem;
            color: var(--takal-muted);
            margin-bottom: 1.3rem;
          }}

          /* --- METRIC CARDS --- */
          .metric-card {{
            background: var(--takal-white);
            border: 1px solid var(--takal-border);
            border-radius: 20px;
            padding: 1.4rem 1.5rem;
            min-height: 148px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: background-color 250ms cubic-bezier(0.4, 0, 0.2, 1),
                        border-color 250ms cubic-bezier(0.4, 0, 0.2, 1),
                        box-shadow 250ms cubic-bezier(0.4, 0, 0.2, 1),
                        transform 250ms cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            cursor: default;
          }}

          .metric-card-header {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.5rem;
          }}

          .metric-label {{
            color: var(--takal-muted);
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            line-height: 1.3;
            transition: color 250ms ease;
          }}

          .metric-header-right {{
            display: flex;
            align-items: center;
            gap: 0.45rem;
          }}

          .metric-badge {{
            background-color: var(--takal-mint-light);
            color: var(--takal-mint-text);
            font-size: 0.72rem;
            font-weight: 600;
            padding: 3px 9px;
            border-radius: 9999px;
            letter-spacing: -0.01em;
            transition: background-color 250ms ease, color 250ms ease;
          }}

          .metric-info {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.15rem;
            height: 1.15rem;
            border: 1px solid #CFD8D3;
            border-radius: 50%;
            color: var(--takal-muted);
            font-size: 0.7rem;
            font-weight: 600;
            cursor: help;
            position: relative;
            transition: border-color 250ms ease, color 250ms ease;
          }}

          .metric-info::after {{
            content: attr(data-tooltip);
            position: absolute;
            z-index: 60;
            left: 50%;
            bottom: calc(100% + 0.6rem);
            width: 220px;
            padding: 0.65rem 0.85rem;
            border-radius: 10px;
            background: #0B221B;
            color: #FFFFFF;
            font-size: 0.76rem;
            font-weight: 400;
            line-height: 1.4;
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
            opacity: 0;
            pointer-events: none;
            transform: translate(-50%, 4px);
            transition: opacity 150ms ease, transform 150ms ease;
          }}

          .metric-info:hover::after {{
            opacity: 1;
            transform: translate(-50%, 0);
          }}

          .metric-value {{
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.035em;
            color: var(--takal-text);
            line-height: 1.15;
            margin: 0.6rem 0 0.25rem 0;
            transition: color 250ms ease;
          }}

          .metric-detail {{
            color: var(--takal-muted);
            font-size: 0.84rem;
            font-weight: 500;
            transition: color 250ms ease;
          }}

          .metric-card:hover,
          .metric-card:focus-within {{
            background-color: var(--takal-forest-card) !important;
            border-color: var(--takal-forest-card) !important;
            transform: translateY(-3px);
            box-shadow: 0 14px 30px rgba(10, 35, 28, 0.20) !important;
          }}

          .metric-card:hover .metric-label,
          .metric-card:focus-within .metric-label {{
            color: #BFE3D6 !important;
          }}

          .metric-card:hover .metric-value,
          .metric-card:focus-within .metric-value {{
            color: #FFFFFF !important;
          }}

          .metric-card:hover .metric-detail,
          .metric-card:focus-within .metric-detail {{
            color: #A8D3C3 !important;
          }}

          .metric-card:hover .metric-badge,
          .metric-card:focus-within .metric-badge {{
            background-color: var(--takal-mint) !important;
            color: #0A231C !important;
          }}

          .metric-card:hover .metric-info,
          .metric-card:focus-within .metric-info {{
            border-color: #4C7A6B !important;
            color: #BFE3D6 !important;
          }}

          @media (prefers-reduced-motion: reduce) {{
            .metric-card,
            .metric-card * {{
              transition: none !important;
            }}
            .metric-card:hover,
            .metric-card:focus-within {{
              transform: none;
            }}
          }}

          /* --- MAIN-AREA FORM FIELDS --- */
          [data-testid="stMain"] [data-testid="stDateInputField"],
          [data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"],
          [data-testid="stMain"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child,
          [data-testid="stMain"] [data-baseweb="input"],
          [data-testid="stMain"] [data-baseweb="base-input"] {{
            background: #FFFFFF !important;
            border: 1px solid #D5E0DA !important;
            border-radius: 12px !important;
            box-shadow: none !important;
            color: var(--takal-text) !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease !important;
          }}

          [data-testid="stMain"] [data-testid="stDateInputField"]:hover,
          [data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"]:hover {{
            border-color: #A9C3B8 !important;
            background: #FBFDFC !important;
          }}

          [data-testid="stMain"] [data-testid="stDateInputField"]:focus-within,
          [data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"]:focus-within,
          [data-testid="stMain"] [data-baseweb="input"]:focus-within {{
            border-color: var(--takal-mint) !important;
            box-shadow: 0 0 0 3px rgba(37, 201, 152, 0.16) !important;
          }}

          [data-testid="stMain"] [data-testid="stDateInputField"] [role="spinbutton"],
          [data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox input,
          [data-testid="stMain"] [data-baseweb="input"] input,
          [data-testid="stMain"] [data-baseweb="select"] span {{
            color: var(--takal-text) !important;
            -webkit-text-fill-color: var(--takal-text) !important;
            background: transparent !important;
            font-size: 0.94rem !important;
            font-weight: 500 !important;
            opacity: 1 !important;
          }}

          [data-testid="stMain"] [data-testid="stDateInputField"] [role="spinbutton"][aria-valuetext="Empty"] {{
            color: var(--takal-muted) !important;
            -webkit-text-fill-color: var(--takal-muted) !important;
          }}

          [data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox button,
          [data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox button svg,
          [data-testid="stMain"] [data-testid="stDateInput"] svg {{
            color: var(--takal-muted) !important;
            fill: var(--takal-muted) !important;
            background: transparent !important;
          }}

          [data-testid="stMain"] [data-testid="stWidgetLabel"] p {{
            color: var(--takal-text) !important;
            font-size: 0.84rem !important;
            font-weight: 600 !important;
          }}

          /* Expander header */
          [data-testid="stExpander"] details {{
            border: 1px solid var(--takal-border) !important;
            border-radius: 14px !important;
            overflow: hidden;
            background: var(--takal-white) !important;
          }}
          [data-testid="stExpander"] summary {{
            background: #F3F8F5 !important;
            color: var(--takal-text) !important;
            font-weight: 600 !important;
            transition: background-color 200ms ease !important;
          }}
          [data-testid="stExpander"] summary:hover {{
            background: #E7F1EC !important;
          }}
          [data-testid="stExpander"] summary * {{
            color: var(--takal-text) !important;
          }}

          /* --- TABLE HEADERS & WRAPPER --- */
          table.weekly-table th,
          [data-testid="stTable"] thead th,
          [data-testid="stMarkdownContainer"] table thead th {{
            background: #0D2E24 !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            letter-spacing: 0.05em;
            border-bottom: 1px solid #1B473A !important;
          }}
          [data-testid="stTable"] tbody td,
          [data-testid="stMarkdownContainer"] table tbody td {{
            color: var(--takal-text) !important;
          }}
          [data-testid="stTable"] table,
          [data-testid="stMarkdownContainer"] table {{
            border-collapse: collapse;
            border-radius: 14px;
            overflow: hidden;
          }}

          /* --- TECHNICAL CARDS & BENCHMARKING TABLES --- */
          .technical-card,
          div[data-testid="stMetric"] {{
            background: var(--takal-white) !important;
            border: 1px solid var(--takal-border) !important;
            border-radius: 16px !important;
            padding: 1.3rem 1.4rem !important;
            min-height: 110px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02) !important;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            cursor: default;
            transition: background-color 250ms cubic-bezier(0.4, 0, 0.2, 1),
                        border-color 250ms cubic-bezier(0.4, 0, 0.2, 1),
                        box-shadow 250ms cubic-bezier(0.4, 0, 0.2, 1),
                        transform 250ms cubic-bezier(0.4, 0, 0.2, 1);
          }}

          .technical-card *,
          div[data-testid="stMetric"] * {{
            transition: color 250ms cubic-bezier(0.4, 0, 0.2, 1);
          }}

          div[data-testid="stMetric"] label {{
            color: var(--takal-muted) !important;
            font-size: 0.74rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
          }}
          div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            color: var(--takal-text) !important;
            font-size: 1.7rem !important;
            font-weight: 700 !important;
          }}

          .technical-card-label {{
            color: var(--takal-muted);
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
          }}
          .technical-card-value {{
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--takal-text);
            margin: 0.35rem 0 0.15rem 0;
            letter-spacing: -0.02em;
          }}
          .technical-card-detail {{
            color: var(--takal-muted);
            font-size: 0.8rem;
          }}

          .technical-card:hover,
          .technical-card:focus-within,
          div[data-testid="stMetric"]:hover {{
            background: var(--takal-forest-card) !important;
            border-color: var(--takal-forest-card) !important;
            transform: translateY(-3px);
            box-shadow: 0 14px 30px rgba(10, 35, 28, 0.20) !important;
          }}
          .technical-card:hover .technical-card-label,
          .technical-card:focus-within .technical-card-label,
          div[data-testid="stMetric"]:hover label,
          div[data-testid="stMetric"]:hover label * {{
            color: #BFE3D6 !important;
          }}
          .technical-card:hover .technical-card-value,
          .technical-card:focus-within .technical-card-value,
          div[data-testid="stMetric"]:hover div[data-testid="stMetricValue"],
          div[data-testid="stMetric"]:hover div[data-testid="stMetricValue"] * {{
            color: #FFFFFF !important;
          }}
          .technical-card:hover .technical-card-detail,
          .technical-card:focus-within .technical-card-detail {{
            color: #A8D3C3 !important;
          }}

          @media (prefers-reduced-motion: reduce) {{
            .technical-card,
            div[data-testid="stMetric"] {{
              transition: none !important;
            }}
            .technical-card:hover,
            div[data-testid="stMetric"]:hover {{
              transform: none;
            }}
          }}

          .result-panel {{
            border-left: 4px solid var(--takal-mint) !important;
            background: var(--takal-white) !important;
            border: 1px solid var(--takal-border);
            border-radius: 16px;
            padding: 1.3rem 1.5rem;
            margin-top: 1.2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
            color: var(--takal-text);
          }}

          div[data-testid="stDataFrame"] {{
            border: 1px solid var(--takal-border) !important;
            border-radius: 16px !important;
            background: var(--takal-white) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02) !important;
            overflow: hidden !important;
          }}

          .chart-card-wrapper {{
            background: var(--takal-white);
            border: 1px solid var(--takal-border);
            border-radius: 20px;
            padding: 1.8rem 1.8rem 1.4rem 1.8rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
            margin-top: 1rem;
          }}

          /* Weekly Table */
          .weekly-table-wrap {{
            overflow-x: auto;
            border: 1px solid var(--takal-border);
            border-radius: 16px;
            background: var(--takal-white);
            margin-bottom: 1.2rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
          }}
          table.weekly-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 0 !important;
            border: none !important;
          }}
          table.weekly-table th {{
            background: #0D2E24;
            color: #FFFFFF;
            font-size: 0.74rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 1rem 1.3rem;
            border: 1px solid #1B473A !important;
            border-top: none !important;
          }}
          table.weekly-table td {{
            padding: 0.95rem 1.3rem;
            font-size: 0.92rem;
            color: var(--takal-text);
            vertical-align: middle;
            white-space: nowrap;
            border: 1px solid var(--takal-border) !important;
            font-variant-numeric: tabular-nums;
          }}
          table.weekly-table tbody tr:hover {{
            background-color: #F1F7F4;
          }}
          table.weekly-table th:first-child,
          table.weekly-table td:first-child {{ border-left: none !important; }}
          table.weekly-table th:last-child,
          table.weekly-table td:last-child {{ border-right: none !important; }}
          table.weekly-table tbody tr:last-child td {{ border-bottom: none !important; }}
          table.weekly-table th.col-left,
          table.weekly-table td.col-left {{ text-align: left !important; }}
          table.weekly-table th.col-center,
          table.weekly-table td.col-center {{ text-align: center !important; }}
          table.weekly-table th.col-right,
          table.weekly-table td.col-right {{ text-align: right !important; }}
          table.weekly-table td.col-label {{
            font-weight: 600;
            color: #0A231C;
          }}
          table.weekly-table td.effect-up {{ color: #0B8561 !important; font-weight: 600; }}
          table.weekly-table td.effect-down {{ color: #B4233F !important; font-weight: 600; }}
          .weekly-forecast {{
            color: #0A231C;
            font-weight: 600;
          }}
          .weekly-actual {{
            color: #10B981;
            font-weight: 600;
          }}
          .weekly-not-yet {{
            color: var(--takal-muted);
            font-style: italic;
          }}

          [data-testid="stExpander"] {{
            border: 1px solid var(--takal-border);
            border-radius: 14px;
            background: var(--takal-white);
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    detail: str,
    tooltip: str,
    badge: str | None = None,
    always_dark: bool = False,
) -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_detail = html.escape(detail)
    safe_tooltip = html.escape(tooltip, quote=True)
    badge_html = f'<span class="metric-badge">{html.escape(badge)}</span>' if badge else ""

    st.markdown(
        f"""
        <section class="metric-card" aria-label="{safe_label}: {safe_value}">
          <div class="metric-card-header">
            <span class="metric-label">{safe_label}</span>
            <div class="metric-header-right">
              {badge_html}
              <span class="metric-info" tabindex="0" role="img" aria-label="About {safe_label}" data-tooltip="{safe_tooltip}">i</span>
            </div>
          </div>
          <div class="metric-value">{safe_value}</div>
          <div class="metric-detail">{safe_detail}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_table(df, align=None, formats=None, label_columns=None, effect_column=None) -> None:
    """Render a DataFrame as the dashboard's standard themed HTML table."""
    align = dict(align or {})
    formats = dict(formats or {})
    label_columns = set(label_columns or [])
    columns = list(df.columns)

    def column_align(column, index):
        if column in align:
            return align[column]
        return "center"

    def fmt(column, value):
        spec = formats.get(column)
        if callable(spec):
            return spec(value)
        if spec:
            return format(value, spec)
        if isinstance(value, float):
            return f"{value:,.3f}"
        return str(value)

    head = "".join(
        f'<th scope="col" class="col-{column_align(c, i)}">{html.escape(str(c))}</th>'
        for i, c in enumerate(columns)
    )
    body = []
    for _, row in df.iterrows():
        cells = []
        for i, c in enumerate(columns):
            classes = [f"col-{column_align(c, i)}"]
            if c in label_columns:
                classes.append("col-label")
            text = fmt(c, row[c])
            if effect_column and c == effect_column:
                lowered = text.lower()
                if lowered.startswith("raises"):
                    classes.append("effect-up")
                elif lowered.startswith("lowers"):
                    classes.append("effect-down")
            cells.append(f'<td class="{" ".join(classes)}">{html.escape(text)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")

    st.markdown(
        '<div class="weekly-table-wrap"><table class="weekly-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def technical_card(label: str, value: str, detail: str) -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_detail = html.escape(detail)
    st.markdown(
        f"""
        <div class="technical-card">
          <div class="technical-card-label">{safe_label}</div>
          <div class="technical-card-value">{safe_value}</div>
          <div class="technical-card-detail">{safe_detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chart(placeholder, figure: go.Figure, key: str) -> None:
    placeholder.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "responsive": True},
        key=key,
    )


def chart_layout(figure: go.Figure, height: int = 440) -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"t": 40, "r": 25, "b": 30, "l": 20},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font={
            "family": "Plus Jakarta Sans, sans-serif",
            "color": "#6C827A",
            "size": 12,
        },
        hoverlabel={
            "bgcolor": "#0A231C",
            "font_color": "#FFFFFF",
            "font_family": "Plus Jakarta Sans, sans-serif",
            "font_size": 12,
        },
    )
    figure.update_xaxes(
        gridcolor="#EDF2EE",
        zeroline=False,
        linecolor="#E3E9E5",
        tickfont={"size": 11, "color": "#758A83"},
    )
    figure.update_yaxes(
        gridcolor="#EDF2EE",
        zeroline=False,
        linecolor="#E3E9E5",
        tickfont={"size": 11, "color": "#758A83"},
    )
    return figure