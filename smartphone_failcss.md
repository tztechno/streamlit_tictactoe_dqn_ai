
```
オリジナル：PCでは表示は良好、スマホは横向きにして立ち上げると良好な表示
st.markdown("""
    <style>
    div[data-testid="column"] {
        width: fit-content !important;
        flex: unset;
    }
    div[data-testid="stHorizontalBlock"] {
        width: fit-content !important;
        margin: auto;
    }
    .stButton button {
        width: 50px !important;
        height: 50px !important;
        font-size: 24px !important;
        font-weight: bold !important;
        padding: 0px !important;
    }
    </style>
""", unsafe_allow_html=True)
```
