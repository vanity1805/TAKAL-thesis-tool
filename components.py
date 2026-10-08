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
          /* Top padding only: shrinks the gap above the title, leaves width and side padding alone */
          [data-testid="stMainBlockContainer"] {{
            padding-top: 4.00rem !important;
          }}

          /* The injected <style> block renders as an empty element that adds a flex gap */
          [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] style),
          [data-testid="stElementContainer"]:has(style) {{
            display: none !important;
          }}

          /* --- SIDEBAR CONTAINER --- */
          [data-testid="stSidebar"] {{
            background-color: var(--takal-forest-dark) !important;
            border-right: none !important;
          }}
          
          [data-testid="stSidebar"] > div {{
            background-color: var(--takal-forest-dark) !important;
          }}

          /* Sidebar width: extra room so the dropdown label is not truncated.
             Scoped to the expanded state so Streamlit's collapse still reclaims the space. */
          [data-testid="stSidebar"][aria-expanded="true"] {{
            min-width: min(22rem, 86vw) !important;
            max-width: min(22rem, 86vw) !important;
          }}
          [data-testid="stSidebar"][aria-expanded="false"] {{
            min-width: 0 !important;
            max-width: 0 !important;
            width: 0 !important;
            margin-left: 0 !important;
            overflow: hidden !important;
            border: none !important;
          }}

          [data-testid="stSidebar"] * {{
            font-family: 'Plus Jakarta Sans', sans-serif !important;
          }}

          /* Hide scrollbars safely without breaking native scrolling */
          [data-testid="stSidebarContent"]::-webkit-scrollbar {{
            display: none !important;
          }}
          [data-testid="stSidebarContent"] {{
            scrollbar-width: none !important;
          }}

          /* Make Streamlit's top header transparent to remove the white gap */
          [data-testid="stHeader"] {{
            background: transparent !important;
          }}

          /* Safely push toggle to bottom using Flexbox on inner containers */
          [data-testid="stSidebarContent"],
          [data-testid="stSidebarUserContent"],
          [data-testid="stSidebarUserContent"] > div,
          [data-testid="stSidebarUserContent"] > div > div[data-testid="stVerticalBlock"] {{
            display: flex !important;
            flex-direction: column !important;
            flex-grow: 1 !important;
          }}
          
          [data-testid="stSidebarUserContent"] {{
            padding-top: 0.5rem !important;
            padding-bottom: 1.25rem !important;
            padding-left: 1.4rem !important;
            padding-right: 1.4rem !important;
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
            --logo-gap-top: 1.75rem;
            --logo-gap-bottom: 0.25rem;
            display: flex !important;
            justify-content: center !important;
            margin: var(--logo-gap-top) 0 var(--logo-gap-bottom) 0 !important;
          }}
          .st-key-home_reset > div,
          .st-key-home_reset [data-testid="stButton"] {{
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
          }}
          .st-key-home_reset button {{
            min-height: 10.5rem !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            background: transparent !important;
            border: 1px solid transparent !important;
            box-shadow: none !important;
            transition: none !important;
            padding: 0 !important;
          }}
          .st-key-home_reset button:hover,
          .st-key-home_reset button:focus,
          .st-key-home_reset button:active {{
            background: transparent !important;
            border-color: transparent !important;
            box-shadow: none !important;
            transform: none !important;
          }}
          .st-key-home_reset button::before {{
            content: "";
            display: block;
            width: 86%;
            height: 9.5rem;
            margin: 0 auto;
            background: center / contain no-repeat url("data:image/png;base64,{brand_logo_b64}");
            filter: brightness(1.05) contrast(1.03);
          }}
          .st-key-home_reset button p {{
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
          }}

          /* --- SIDEBAR DROPDOWN SPECIFICATION --- */
          [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
          [data-testid="stSidebar"] [data-testid="stSelectbox"] label p {{
            color: #7B9B8E !important;
            font-size: 0.95rem !important;
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
            overflow: hidden !important;
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

          div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])) {{
            background: transparent !important;
          }}
          div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])) [role="listbox"],
          div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])) ul {{
            background-color: #0D2E24 !important;
            background: #0D2E24 !important;
            border: 1px solid #1B473A !important;
            border-radius: 16px !important;
            padding: 6px !important;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.55) !important;
          }}
          div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])) [role="option"],
          div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])) li {{
            background-color: transparent !important;
            color: #B4CDC3 !important;
            font-size: 0.92rem !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            cursor: pointer !important;
          }}
          div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])) [role="option"]:hover,
          div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])) [role="option"][aria-selected="true"],
          div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])) li:hover,
          div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])) li[aria-selected="true"] {{
            background-color: #174537 !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
          }}

          /* Date picker popup: solid white surface */
          div[data-baseweb="popover"]:has([data-baseweb="calendar"]),
          div[data-baseweb="popover"]:has([data-baseweb="calendar"]) > div {{
            background: #FFFFFF !important;
            border-radius: 12px !important;
            box-shadow: 0 12px 30px rgba(10, 35, 28, 0.14) !important;
          }}

          div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])),
          div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])) > div,
          div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])) > div > div {{
            background: transparent !important;
            border-radius: 16px !important;
            box-shadow: none !important;
            border: none !important;
          }}

          /* Dropdown popover: clear every white wrapper behind the rounded dark list.
          :where() keeps specificity at zero so the dark list rules above still win. */
          :where(div[data-baseweb="popover"]:has([role="listbox"]):not(:has([data-testid="stMainMenuList"])):not(:has([data-baseweb="calendar"])) div) {{
            background: transparent !important;
            border-color: transparent !important;
            box-shadow: none !important;
          }}

          /* Select dropdown popup: clear every wrapper behind the rounded dark list */
          div[data-baseweb="popover"]:has([data-testid="stSelectboxVirtualDropdown"]),
          div[data-baseweb="popover"]:has([data-testid="stSelectboxVirtualDropdown"]) div:not([data-testid="stSelectboxVirtualDropdown"]):not([data-testid="stSelectboxVirtualDropdown"] *) {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            border-radius: 16px !important;
          }}

          div[data-baseweb="popover"]:has([data-baseweb="calendar"]),
          div[data-baseweb="popover"]:has([data-baseweb="calendar"]) > div {{
            background: #FFFFFF !important;
            border-radius: 12px !important;
            box-shadow: 0 12px 30px rgba(10, 35, 28, 0.14) !important;
          }}

          div[data-baseweb="calendar"] [role="gridcell"]:not([aria-selected="true"]):hover::after {{
            background-color: #DDEFE8 !important;
          }}
          div[data-baseweb="calendar"] [role="gridcell"]:not([aria-selected="true"]):hover {{
            color: var(--takal-text) !important;
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

          [data-testid="stSidebar"] .st-key-technical_details label > div:first-child {{
            flex: 0 0 auto !important;
            background-color: rgba(255, 255, 255, 0.16) !important;
            border-color: #7FA394 !important;
            box-shadow: inset 0 0 0 1.5px #7FA394 !important;
            transition: background-color 200ms ease, box-shadow 200ms ease !important;
          }}
          [data-testid="stSidebar"] .st-key-technical_details label:hover > div:first-child {{
            background-color: rgba(255, 255, 255, 0.24) !important;
          }}
          [data-testid="stSidebar"] .st-key-technical_details label > div:first-child > div {{
            background-color: #E8F5EF !important;
          }}
          [data-testid="stSidebar"] .st-key-technical_details label:has(input:checked) > div:first-child {{
            background-color: var(--takal-mint) !important;
            border-color: var(--takal-mint) !important;
            box-shadow: none !important;
          }}

          /* Safety net: the label text never gets a background */
          [data-testid="stSidebar"] .st-key-technical_details label > div:not(:first-child),
          [data-testid="stSidebar"] .st-key-technical_details label > div:not(:first-child) * {{
            background: transparent !important;
            box-shadow: none !important;
          }}

                  /* --- PILL TABS SITTING ON A THIN LINE (ARIA roles only) --- */
          [data-baseweb="tab-highlight"],
          [data-baseweb="tab-border"] {{
            display: none !important;
          }}
          [role="tablist"]::before,
          [role="tablist"]::after {{
            content: none !important;
            display: none !important;
          }}

          /* Pill container: rounded on top, square bottom corners, sitting on the line */
          [role="tablist"] {{
            background-color: #E6EBE8 !important;
            padding: 5px 5px 0 5px !important; 
            border-radius: 16px 16px 0 0 !important;
            gap: 4px !important;
            margin: 0 !important;
            width: fit-content !important;
            border-bottom: none !important;
            box-shadow: none !important;
          }}

          /* The thin line is the top edge of the panel, directly under the container */
          [role="tabpanel"] {{
            margin-top: 0 !important;
            padding-top: 1.5rem !important;
            border-top: 1px solid #E3E9E5 !important;
          }}
          [role="tabpanel"] [role="tabpanel"] {{
            padding-top: 1.1rem !important;
          }}

          button[role="tab"] {{
            border-radius: 12px 12px 0 0 !important;   /* was: 12px */
            padding: 8px 20px !important;
            height: auto !important;
            background: transparent !important;
            border: none !important;
            outline: none !important;
            color: #556E65 !important;
            font-size: 1.05rem !important;
            font-weight: 600 !important;
          }}
          
          button[role="tab"]:hover {{
            color: var(--takal-text) !important;
          }}
          button[role="tab"] p,
          button[role="tab"] [data-testid="stMarkdownContainer"] p {{
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            margin: 0 !important;
          }}
          button[role="tab"][aria-selected="true"] {{
            background-color: #FFFFFF !important;
            color: var(--takal-text) !important;
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
            max-width: 100%;
          }}

          .page-intro strong {{
            color: var(--takal-text);
            font-weight: 600;
          }}

          [data-testid="stMain"] hr {{
            margin: 1.25rem 0 !important;
          }}
          [data-testid="stMain"] [data-testid="stElementContainer"]:has(hr) + [data-testid="stElementContainer"] h3 {{
            margin-top: 0 !important;
            padding-top: 0 !important;
          }}

          /* st.subheader headings match the custom section headings */
          [data-testid="stMain"] h3 {{
            font-size: 1.9rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.025em !important;
            color: var(--takal-text) !important;
            margin-top: 2rem !important;
          }}

          /* A heading that directly follows a divider (hr or markdown divider) needs no extra top margin */
          [data-testid="stMain"] [data-testid="stElementContainer"]:has(hr) + [data-testid="stElementContainer"] .section-subheading,
          [data-testid="stMain"] [data-testid="stElementContainer"]:has(hr) + [data-testid="stElementContainer"] h3 {{
            margin-top: 0 !important;
            padding-top: 0 !important;
          }}

          .section-subheading {{
            font-size: 1.9rem;
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
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
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
            right: 0;
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
            transform: translateY(4px);
            transition: opacity 150ms ease, transform 150ms ease;
          }}

          .metric-info:hover::after {{
            opacity: 1;
            transform: translateY(0);
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
            z-index: 30;
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

          /* --- MAIN-AREA STRICT DROPDOWNS --- */
          [data-testid="stMain"] [data-testid="stWidgetLabel"] p {{
            color: var(--takal-text) !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
          }}
          
          /* Force pointer cursor on the entire selectbox container to remove the I-beam */
          [data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"],
          [data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox > [role="group"] *,
          [data-testid="stMain"] [data-testid="stSelectbox"] div[data-baseweb="select"],
          [data-testid="stMain"] [data-testid="stSelectbox"] div[data-baseweb="select"] * {{
            cursor: pointer !important;
          }}

          /* Disable text selection and typing caret on inputs */
          [data-testid="stMain"] [data-testid="stSelectbox"] .react-aria-ComboBox input,
          [data-testid="stMain"] [data-testid="stSelectbox"] div[data-baseweb="select"] input {{
            caret-color: transparent !important;
            user-select: none !important;
            pointer-events: none !important;
          }}

          [data-testid="stMain"] .takal-select-label {{
            color: var(--takal-text) !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            margin-bottom: 0.25rem !important;
          }}

          [data-testid="stMain"] .st-key-overview_horizon_popover button[data-testid="stPopoverButton"],
          [data-testid="stMain"] .st-key-overview_date_order_popover button[data-testid="stPopoverButton"],
          [data-testid="stMain"] .st-key-overview_rows_per_page_popover button[data-testid="stPopoverButton"] {{
            min-height: 40px !important;
            border: 1px solid #D5E0DA !important;
            border-radius: 12px !important;
            background: #FFFFFF !important;
            color: var(--takal-text) !important;
            box-shadow: none !important;
            cursor: pointer !important;
          }}

          [data-testid="stMain"] .st-key-overview_horizon_popover button[data-testid="stPopoverButton"]:hover,
          [data-testid="stMain"] .st-key-overview_date_order_popover button[data-testid="stPopoverButton"]:hover,
          [data-testid="stMain"] .st-key-overview_rows_per_page_popover button[data-testid="stPopoverButton"]:hover {{
            border-color: #A9C3B8 !important;
            background: #FBFDFC !important;
          }}

          [data-testid="stMain"] .st-key-overview_horizon_popover button[data-testid="stPopoverButton"]:focus,
          [data-testid="stMain"] .st-key-overview_date_order_popover button[data-testid="stPopoverButton"]:focus,
          [data-testid="stMain"] .st-key-overview_rows_per_page_popover button[data-testid="stPopoverButton"]:focus {{
            border-color: var(--takal-mint) !important;
            box-shadow: 0 0 0 3px rgba(37, 201, 152, 0.16) !important;
          }}

          [data-testid="stMain"] .st-key-overview_horizon_popover button[data-testid="stPopoverButton"] svg,
          [data-testid="stMain"] .st-key-overview_date_order_popover button[data-testid="stPopoverButton"] svg,
          [data-testid="stMain"] .st-key-overview_rows_per_page_popover button[data-testid="stPopoverButton"] svg {{
            fill: var(--takal-muted) !important;
          }}

          [data-testid="stMain"] [data-testid="stPopoverBody"] {{
            background: #0D2E24 !important;
            border: 1px solid #1B473A !important;
            border-radius: 12px !important;
            padding: 6px !important;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.28) !important;
          }}

          [data-testid="stMain"] [data-testid="stPopoverBody"] button[data-testid="stBaseButton-tertiary"] {{
            min-height: 36px !important;
            border: 0 !important;
            border-radius: 8px !important;
            background: transparent !important;
            color: #B4CDC3 !important;
            box-shadow: none !important;
          }}

          [data-testid="stMain"] [data-testid="stPopoverBody"] button[data-testid="stBaseButton-tertiary"]:hover {{
            background: #174537 !important;
            color: #FFFFFF !important;
          }}

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

          /* Date input: keep the border on the field only, not the nested wrappers */
          [data-testid="stMain"] [data-testid="stDateInput"] [data-baseweb="input"],
          [data-testid="stMain"] [data-testid="stDateInput"] [data-baseweb="base-input"] {{
            border: none !important;
            box-shadow: none !important;
            background: transparent !important;
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
            font-size: 0.85rem !important;
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
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
          }}
          /* Greek letters must stay lowercase: uppercase turns α into a Latin-looking "A" */
          .technical-card-label .no-upper {{
            text-transform: none;
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
            border-left: 4px solid var(--takal-forest-pill) !important;
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

          .st-key-chart_card_forecast {{
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
            font-size: 0.85rem;
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
          table.weekly-table td.effect-down {{ color: #0EA5E9 !important; font-weight: 600; }}
          
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

          /* --- SIDEBAR HIDE / SHOW BUTTON: visible on the dark sidebar --- */
          [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
          [data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{
            opacity: 1 !important;
            visibility: visible !important;
          }}
          [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
          [data-testid="stSidebar"] [data-testid="stSidebarHeader"] button {{
            color: #CFE3DA !important;
            background: transparent !important;
            border: none !important;
            border-radius: 10px !important;
            opacity: 1 !important;
            transition: background-color 0.2s ease !important;
          }}
          [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover,
          [data-testid="stSidebar"] [data-testid="stSidebarHeader"] button:hover {{
            background: rgba(255, 255, 255, 0.20) !important;
          }}
          [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] *,
          [data-testid="stSidebar"] [data-testid="stSidebarHeader"] button * {{
            color: #CFE3DA !important;
            fill: #CFE3DA !important;
          }}
          /* Re-open button shown in the main header while the sidebar is collapsed */
          [data-testid="stExpandSidebarButton"],
          [data-testid="stExpandSidebarButton"] *,
          [data-testid="collapsedControl"],
          [data-testid="collapsedControl"] * {{
            color: var(--takal-text) !important;
            opacity: 1 !important;
          }}

          /* --- EQUAL-SIZE CARDS: every card in a row stretches to the tallest one --- */
          :is([data-testid="stHorizontalBlock"]:has(.metric-card),
              [data-testid="stHorizontalBlock"]:has(.technical-card)) {{
            align-items: stretch !important;
          }}
          :is([data-testid="stHorizontalBlock"]:has(.metric-card),
              [data-testid="stHorizontalBlock"]:has(.technical-card)) > :is([data-testid="stColumn"], [data-testid="column"]) {{
            display: flex !important;
            flex-direction: column !important;
          }}
          :is([data-testid="stHorizontalBlock"]:has(.metric-card),
              [data-testid="stHorizontalBlock"]:has(.technical-card)) > :is([data-testid="stColumn"], [data-testid="column"]) > [data-testid="stVerticalBlock"] {{
            flex: 1 1 auto !important;
          }}
          /* Pass the stretch down through Streamlit's wrapper elements to the card itself */
          :is([data-testid="stHorizontalBlock"]:has(.metric-card),
              [data-testid="stHorizontalBlock"]:has(.technical-card)) [data-testid="stElementContainer"],
          :is([data-testid="stHorizontalBlock"]:has(.metric-card),
              [data-testid="stHorizontalBlock"]:has(.technical-card)) [data-testid="stMarkdown"],
          :is([data-testid="stHorizontalBlock"]:has(.metric-card),
              [data-testid="stHorizontalBlock"]:has(.technical-card)) [data-testid="stMarkdownContainer"] {{
            display: flex !important;
            flex-direction: column !important;
            flex: 1 1 auto !important;
          }}
          .metric-card,
          .technical-card {{
            flex: 1 1 auto;
          }}
          /* Overview cards: reserve two label lines so values and details line up across cards */
          .metric-card {{
            justify-content: flex-start;
          }}
          .metric-card-header {{
            min-height: 3.4rem;
          }}
          .metric-card .metric-detail {{
            margin-top: auto;
          }}
          @media (max-width: 640px) {{
            .metric-card-header {{
              min-height: 0;
            }}
          }}

          /* Badge such as "+ ₱1.48" stays on one line; the label wraps instead */
          .metric-badge {{
            white-space: nowrap;
          }}
          .metric-header-right {{
            flex-shrink: 0;
          }}
          .metric-label {{
            min-width: 0;
          }}

          /* ================= RESPONSIVE LAYOUT ================= */

          /* Fluid headings: scale smoothly between phone and desktop */
          .page-heading {{
            font-size: clamp(1.7rem, 0.9rem + 3vw, 2.35rem) !important;
          }}
          .section-subheading,
          [data-testid="stMain"] h3 {{
            font-size: clamp(1.3rem, 0.8rem + 2vw, 1.9rem) !important;
          }}
          .metric-value {{
            font-size: clamp(1.5rem, 1rem + 1.4vw, 2.1rem);
          }}
          .technical-card-value {{
            font-size: clamp(1.3rem, 0.9rem + 1.2vw, 1.65rem);
            overflow-wrap: anywhere;
          }}

          /* Tab pills scroll sideways instead of overflowing the page */
          @media (max-width: 900px) {{
            [role="tablist"] {{
              max-width: 100% !important;
              overflow-x: auto !important;
              overflow-y: hidden !important;
              scrollbar-width: none;
            }}
            [role="tablist"]::-webkit-scrollbar {{
              display: none;
            }}
          }}
          button[role="tab"] {{
            flex-shrink: 0 !important;
            white-space: nowrap !important;
          }}

          .metric-info::after {{
            width: min(220px, 70vw);
          }}

          /* Tablet and below: 4-up / 3-up card rows wrap into 2 per row */
          @media (max-width: 1100px) {{
            [data-testid="stHorizontalBlock"]:has(.metric-card),
            [data-testid="stHorizontalBlock"]:has(.technical-card) {{
              flex-wrap: wrap !important;
              row-gap: 1rem !important;
            }}
            [data-testid="stHorizontalBlock"]:has(.metric-card) > [data-testid="stColumn"],
            [data-testid="stHorizontalBlock"]:has(.metric-card) > [data-testid="column"],
            [data-testid="stHorizontalBlock"]:has(.technical-card) > [data-testid="stColumn"],
            [data-testid="stHorizontalBlock"]:has(.technical-card) > [data-testid="column"] {{
              flex: 1 1 calc(50% - 1rem) !important;
              min-width: calc(50% - 1rem) !important;
            }}
          }}

          @media (max-width: 900px) {{
            [data-testid="stMainBlockContainer"],
            [data-testid="stAppViewContainer"] .main .block-container {{
              padding: 2.75rem 1.4rem 2.5rem 1.4rem !important;
            }}
            .page-intro {{
              margin-bottom: 1.4rem;
            }}
            .result-panel {{
              padding: 1.1rem 1.2rem;
            }}
          }}

          /* Phones */
          @media (max-width: 640px) {{
            [data-testid="stMainBlockContainer"],
            [data-testid="stAppViewContainer"] .main .block-container {{
              padding: 2.5rem 0.85rem 2rem 0.85rem !important;
            }}
            [data-testid="stHorizontalBlock"]:has(.metric-card) > [data-testid="stColumn"],
            [data-testid="stHorizontalBlock"]:has(.metric-card) > [data-testid="column"],
            [data-testid="stHorizontalBlock"]:has(.technical-card) > [data-testid="stColumn"],
            [data-testid="stHorizontalBlock"]:has(.technical-card) > [data-testid="column"] {{
              flex: 1 1 100% !important;
              min-width: 100% !important;
            }}
            .metric-card {{
              min-height: 0;
              padding: 1.1rem 1.2rem;
            }}
            .technical-card,
            div[data-testid="stMetric"] {{
              min-height: 0;
              padding: 1rem 1.15rem !important;
            }}
            /* Tighter tab-rule spacing on phones */
            [data-baseweb="tabs"] {{
              --tab-rule-gap-above: 0.65rem;
              --tab-rule-gap-below: 1.1rem;
            }}
            [role="tablist"] {{
              padding: 4px !important;
            }}
            button[role="tab"] {{
              padding: 7px 14px !important;
            }}
            button[role="tab"] p,
            button[role="tab"] [data-testid="stMarkdownContainer"] p {{
              font-size: 0.92rem !important;
            }}
            .st-key-chart_card_forecast {{
              padding: 0.9rem 0.5rem 0.6rem 0.5rem;
              border-radius: 16px;
            }}
            table.weekly-table th {{
              padding: 0.75rem 0.8rem;
              font-size: 0.74rem;
            }}
            table.weekly-table td {{
              padding: 0.7rem 0.8rem;
              font-size: 0.84rem;
            }}
            /* Smaller chart text so plots keep room on narrow screens */
            .js-plotly-plot .xtick text,
            .js-plotly-plot .ytick text,
            .js-plotly-plot .legend text {{
              font-size: 11px !important;
            }}
            .js-plotly-plot .g-xtitle text,
            .js-plotly-plot .g-ytitle text {{
              font-size: 12px !important;
            }}
            .js-plotly-plot .modebar-container {{
              display: none !important;
            }}
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
            "color": "#2F4A41",
            "size": 14,
        },
        hoverlabel={
            "bgcolor": "#0A231C",
            "font_color": "#FFFFFF",
            "font_family": "Plus Jakarta Sans, sans-serif",
            "font_size": 12,
        },
    )
    axis_style = {
        "gridcolor": "#EDF2EE",
        "zeroline": False,
        "linecolor": "#E3E9E5",
        "tickfont": {"size": 13, "color": "#2F4A41"},
        "title_font": {"size": 14, "color": "#2F4A41"},
        "automargin": True,
    }
    figure.update_xaxes(**axis_style)
    figure.update_yaxes(**axis_style)
    return figure