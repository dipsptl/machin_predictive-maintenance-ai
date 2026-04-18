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

        return "Suggested actions: " + ", ".join(suggestions)

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

if user_query:
    response = agent_response(user_query, temp, vib, load, press, result)
    st.success(response)

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
st.write(" ")
st.subheader("💡 Suggestions")

st.write(" ")

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

st.markdown("---")
st.subheader("🤖 Ask AI Agent")

user_query = st.text_input("Ask about machine condition")
