import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# ===== PAGE SETTINGS =====
st.set_page_config(page_title="Predictive Maintenance AI", layout="centered")

# ===== TITLE =====
st.markdown("""
<h1 style='text-align:center; color:#FF8C00;'>
⚙️ Machine Predictive Maintenance AI Dashboard
</h1>
""", unsafe_allow_html=True)

# ===== AI EXPLANATION FUNCTION =====
def ai_explanation(temp, vib, load, press, result):
    if result == 0:
        return "Machine is operating within safe limits."
    else:
        reasons = []
        if temp > 80:
            reasons.append("high temperature")
        if vib > 40:
            reasons.append("high vibration")
        if load > 75:
            reasons.append("high load")
        if press > 60:
            reasons.append("high pressure")

        return "Machine is at risk due to " + ", ".join(reasons)

# ===== AI AGENT =====
def agent_response(query, temp, vib, load, press, result):
    query = query.lower()

    if "why" in query or "reason" in query:
        return ai_explanation(temp, vib, load, press, result)

    elif "safe" in query or "status" in query:
        return "Machine is SAFE" if result == 0 else "Machine is at RISK"

    elif "what to do" in query or "solution" in query:
        suggestions = []

        if temp > 80:
            suggestions.append("Reduce temperature")
        if vib > 40:
            suggestions.append("Check vibration")
        if load > 75:
            suggestions.append("Reduce load")
        if press > 60:
            suggestions.append("Check pressure")

        if suggestions:
            return "Suggested actions: " + ", ".join(suggestions)
        else:
            return "No immediate action required."

    else:
        return "Please ask about machine status, reason, or solution."

# ===== LOAD DATA =====
data = pd.read_csv("machine_data.csv")

X = data[['Temperature', 'Vibration', 'Load', 'Pressure']]
y = data['Failure']

model = LogisticRegression()
model.fit(X, y)

# ===== INPUT SECTION =====
st.subheader("⚙️ Enter Machine Parameters")

temp = st.slider("Temperature", 50, 100)
vib = st.slider("Vibration", 10, 60)
load = st.slider("Load", 40, 100)
press = st.slider("Pressure", 20, 80)

# ===== PREDICTION =====
st.markdown("---")
st.subheader("📊 Prediction Result")

result = model.predict([[temp, vib, load, press]])[0]
prob = model.predict_proba([[temp, vib, load, press]])[0][1]

if result == 0:
    st.success("🟢 Machine Safe")
else:
    st.error("🔴 High Failure Risk")

st.markdown(f"""
<h3 style='color:white;'>
Failure Risk: {prob*100:.1f} %
</h3>
""", unsafe_allow_html=True)

# ===== SUGGESTIONS =====
st.subheader("💡 Suggestions")

if temp > 80:
    st.warning("High temperature – cooling needed")

if vib > 40:
    st.warning("High vibration – check alignment")

if load > 75:
    st.warning("Reduce load")

if press > 60:
    st.warning("Check pressure system")

# ===== GRAPH =====
st.markdown("---")
st.subheader("📈 Analysis")

try:
    fig, ax = plt.subplots()
    ax.scatter(data['Temperature'], data['Vibration'])
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Vibration")

    st.pyplot(fig)
except:
    st.error("Graph not available")

# ===== AI CHAT SECTION =====
st.markdown("---")
st.subheader("🤖 Ask AI Agent")

user_query = st.text_input("Ask about machine condition")

if user_query:
    response = agent_response(user_query, temp, vib, load, press, result)
    st.success(response)
