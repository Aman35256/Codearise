import streamlit as st
import os
import pandas as pd
import re
from groq import Groq
from datetime import datetime

# Set page config
st.set_page_config(page_title="AI Medical Consultancy", layout="wide")

# Load custom CSS
def load_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Please ensure 'style.css' is in the directory.")

load_css()

# Initialize session state variables
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'symptom_details' not in st.session_state:
    st.session_state.symptom_details = []
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}
if 'appointment_details' not in st.session_state:
    st.session_state.appointment_details = None
if 'appointment_summary' not in st.session_state:
    st.session_state.appointment_summary = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

class MedicalAnalysisSystem:
    def __init__(self, dataset_path):
        try:
            self.data = pd.read_csv(dataset_path,encoding='latin1')
            print("Dataset Columns:", self.data.columns.tolist())  # Debug
            print("Sample Data:\n", self.data.head())  # Debug
            
            # Clean data - remove placeholder rows
            self.data = self.data[~self.data['Symptom'].str.contains('Symptom|Condition', case=False)]
            
            self.data['Risk Score'] = pd.to_numeric(self.data['Risk Score'], errors='coerce')
            # Handle missing values
            self.data['Risk Score'] = self.data['Risk Score'].fillna(0)
            
            # Calculate MAX_RISK_SCORE dynamically
            symptom_max_risk = self.data.groupby('Symptom')['Risk Score'].max().sum()
            max_age = 120
            max_age_risk = (max_age - 40) * 0.05 if max_age > 40 else 0
            self.MAX_RISK_SCORE = symptom_max_risk + max_age_risk
            
            self.local_messages = []
            self.severity_mapping = {
                'Mild': ['mild', 'slight', 'minor', 'low grade'],
                'Moderate': ['moderate', 'medium', 'average'],
                'Severe': ['severe', 'high', 'extreme', 'critical', 'intense', 'very bad', 'acute']
            }
            self.negation_words = {'no', 'not', 'denies', 'without', 'negative', 'none', 'denied'}
        except Exception as e:
            st.error(f"Dataset Error: {str(e)}")
            raise

    def add_patient_data(self, patient_message):
        try:
            if not patient_message:
                raise ValueError("Patient message cannot be empty")
            self.local_messages.append({
                'message': patient_message,
                'timestamp': datetime.now().timestamp()
            })
        except Exception as e:
            st.error(f"Error adding patient data: {str(e)}")

    def extract_info_from_bot_response(self, bot_response_data):
        try:
            if not bot_response_data:
                return 0, [], {}

            bot_response_text = str(bot_response_data)
            bot_response_lower = bot_response_text.lower()

            # Age extraction
            age = 0
            age_pattern = r'(\d{1,3})\s*(?:years?-?old|yo|years|-years-old?)'
            age_match = re.search(age_pattern, bot_response_text, re.IGNORECASE)
            if age_match:
                age = int(age_match.group(1))
                if not (0 <= age <= 120): age = 20

            # Symptom extraction
            symptoms = []
            for symptom in self.data['Symptom'].unique():
                symptom_lower = symptom.lower()
                pattern = re.compile(r'\b' + re.escape(symptom_lower) + r'\b', re.IGNORECASE)
                matches = pattern.finditer(bot_response_lower)
                for match in matches:
                    start_pos = match.start()
                    preceding_text = bot_response_lower[:start_pos].split()
                    preceding_words = preceding_text[-3:]
                    if not any(neg in preceding_words for neg in self.negation_words):
                        symptoms.append(symptom)
                        break

            # Severity analysis
            symptom_severity = {}
            for symptom in symptoms:
                symptom_lower = symptom.lower()
                highest_severity_score = 0
                pattern = re.compile(r'\b' + re.escape(symptom_lower) + r'\b', re.IGNORECASE)
                matches = pattern.finditer(bot_response_lower)
                for match in matches:
                    start, end = match.start(), match.end()
                    words = bot_response_lower.split()
                    match_index = len(bot_response_lower[:start].split())
                    context_start = max(0, match_index - 5)
                    context_end = min(len(words), match_index + 6)
                    context = ' '.join(words[context_start:context_end])
                    for severity, keywords in self.severity_mapping.items():
                        for keyword in keywords:
                            if re.search(r'\b' + re.escape(keyword) + r'\b', context):
                                condition_data = self.data[(self.data['Symptom'] == symptom) & 
                                                          (self.data['Condition'] == severity)]
                                if not condition_data.empty:
                                    risk_score = condition_data['Risk Score'].values[0]
                                    if risk_score > highest_severity_score:
                                        highest_severity_score = risk_score
                if highest_severity_score == 0:
                    highest_severity_score = self.data[self.data['Symptom'] == symptom]['Risk Score'].max()
                symptom_severity[symptom] = highest_severity_score

            return age, symptoms, symptom_severity

        except Exception as e:
            st.error(f"Extraction Error: {str(e)}")
            return 0, [], {}

    def calculate_risk_score(self, age, symptoms, symptom_severity):
        try:
            # Validate symptoms
            valid_symptoms = [s for s in symptoms if s in self.data['Symptom'].values]
            if not valid_symptoms:
                return "Unknown", 0, 0
                
            # Calculate scores with validation
            symptom_risk = sum(float(symptom_severity.get(s, 0)) for s in valid_symptoms)
            age_risk = max((age - 40) * 0.05, 0) if age >= 40 else 0
            final_score = symptom_risk + age_risk
            
            # Ensure we don't divide by zero
            max_score = self.MAX_RISK_SCORE if self.MAX_RISK_SCORE > 0 else 1
            risk_pct = min(100, max(0, (final_score / max_score) * 100))
            
            if risk_pct <= 30: label = "Low"
            elif risk_pct <= 70: label = "Medium"
            else: label = "High"
            
            return label, final_score, round(risk_pct, 1)
        except Exception as e:
            st.error(f"Risk Calculation Error: {str(e)}")
            return "Low", 0, 0

    def analyze_patient_data(self, patient_message):
        """Full analysis workflow"""
        try:
            # Clean input message
            patient_message = patient_message.replace("Symptom", "").replace("Condition", "")
            
            self.add_patient_data(patient_message)
            age, symptoms, severity = self.extract_info_from_bot_response(patient_message)
            
            # Filter invalid symptoms
            valid_symptoms = [s for s in symptoms if s in self.data['Symptom'].values]
            if not valid_symptoms:
                return {"error": "No valid symptoms detected"}
            
            # Get unique conditions from valid symptoms
            conditions = self.data[self.data['Symptom'].isin(valid_symptoms)]['Condition'].unique()
            valid_conditions = [c for c in conditions if c not in ['Normal', 'Moderate', 'Severe', 'Condition']]
            
            risk_label, risk_score, risk_pct = self.calculate_risk_score(age, valid_symptoms, severity)
            
            return {
                'age': age,
                'symptoms': valid_symptoms,
                'symptom_severity': severity,
                'risk_label': risk_label,
                'risk_score': round(risk_score, 2),
                'risk_percentage': risk_pct,
                'possible_conditions': valid_conditions,
                'analysis_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Analysis Error: {str(e)}"}

    def process_user_data(self):
        try:
            if not self.local_messages:
                return {"error": "No messages available"}
            
            latest = max(self.local_messages, key=lambda x: x['timestamp'])
            age, symptoms, severity = self.extract_info_from_bot_response(latest['message'])
            
            if not symptoms: return {"error": "No symptoms detected"}
            
            risk_label, risk_score, risk_pct = self.calculate_risk_score(age, symptoms, severity)
            
            return {
                'age': age,
                'symptoms': symptoms,
                'symptom_severity': severity,
                'risk_label': risk_label,
                'risk_score': round(risk_score, 2),
                'risk_percentage': risk_pct,
                'possible_conditions': self.data[self.data['Symptom'].isin(symptoms)]['Condition'].unique().tolist(),
                'analysis_timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": f"Processing Error: {str(e)}"}

def initialize_groq_client():
    try:
        api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        if not api_key:
            api_key = st.text_input("Enter Groq API Key:", type="password")
            if not api_key: return False
        
        st.session_state.client = Groq(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Groq Error: {str(e)}")
        return False

def symptom_interrogation_step():
    client = st.session_state.client
    main_symptom = st.session_state.patient_info['main_symptom']
    step = len(st.session_state.symptom_details)

    if step == 0:
        medical_focus = {
            'pain': "location/radiation/provoking factors",
            'fever': "pattern/associated symptoms/response to meds",
            'gi': "bowel changes/ingestion timing/associated symptoms",
            'respiratory': "exertion relationship/sputum/triggers"
        }
        focus = medical_focus.get(main_symptom.lower(), "temporal pattern/severity progression/associated symptoms")
        prompt = f"""As an ER physician, ask ONE high-yield question about {main_symptom}
        focusing on {focus}. Use simple, patient-friendly language. Ask only ONE question."""
    else:
        last_qa = st.session_state.symptom_details[-1]
        prompt = f"""Based on previous Q: {last_qa['question']} → A: {last_qa['answer']}
        Ask the NEXT critical question about {main_symptom} considering red flags."""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.3
        )
        question = response.choices[0].message.content.strip()
        if not question.endswith('?'): question += '?'
        st.session_state.current_question = question
    except Exception as e:
        st.error(f"Question Generation Error: {str(e)}")
        st.stop()

def handle_symptom_interrogation():
    st.header("Symptom Analysis")
    
    if st.session_state.current_step == 1:
        symptom_interrogation_step()
        st.session_state.current_step = 2
    
    if 'current_question' in st.session_state:
        with st.form("symptom_qna"):
            st.markdown(f'<div class="dr-message">👨‍⚕ {st.session_state.current_question}</div>', unsafe_allow_html=True)
            answer = st.text_input("Your answer:", key=f"answer_{len(st.session_state.symptom_details)}")
            
            if st.form_submit_button("Next"):
                if answer:
                    st.session_state.symptom_details.append({
                        "question": st.session_state.current_question,
                        "answer": answer
                    })
                    del st.session_state.current_question

                    # Emergency check
                    if len(st.session_state.symptom_details) >= 3:
                        last_answer = st.session_state.symptom_details[-1]['answer']
                        try:
                            urgency_check = st.session_state.client.chat.completions.create(
                                messages=[{"role": "user", "content": f"Does this indicate emergency? '{last_answer}' Yes/No"}],
                                model="mixtral-8x7b-32768",
                                temperature=0
                            ).choices[0].message.content
                            if 'YES' in urgency_check.upper():
                                st.markdown('<div class="emergency-alert">🚨 Emergency Detected! Seek Immediate Care.</div>', unsafe_allow_html=True)
                                st.session_state.current_step = 4
                                return
                        except: pass

                    if len(st.session_state.symptom_details) < 7:
                        st.session_state.current_step = 1
                    else:
                        st.session_state.current_step = 3
                    st.rerun()
                else:
                    st.warning("Please provide an answer")

def collect_basic_info():
    st.header("Patient Information")
    with st.form("basic_info"):
        st.session_state.patient_info['name'] = st.text_input("Full Name")
        st.session_state.patient_info['age'] = st.number_input("Age", min_value=0, max_value=120)
        st.session_state.patient_info['gender'] = st.selectbox("Gender", ["Male", "Female", "Other"])
        st.session_state.patient_info['main_symptom'] = st.text_input("Main Symptom")
        
        if st.form_submit_button("Next"):
            if all(st.session_state.patient_info.get(k) for k in ['name', 'age', 'gender', 'main_symptom']):
                st.session_state.current_step = 1
                st.rerun()
            else:
                st.warning("Please fill all fields")

def collect_medical_history():
    st.header("Medical History")
    with st.form("medical_history"):
        st.session_state.patient_info['medical_history'] = st.text_area("Relevant Medical History")
        st.session_state.patient_info['medications'] = st.text_area("Current Medications")
        st.session_state.patient_info['allergies'] = st.text_input("Known Allergies")
        st.session_state.patient_info['last_meal'] = st.text_input("Last Meal Time")
        st.session_state.patient_info['recent_travel'] = st.text_input("Recent Travel History")
        
        if st.form_submit_button("Submit"):
            st.session_state.current_step = 4
            st.rerun()

def generate_risk_assessment():
    st.header("Comprehensive Assessment")
    
    try:
        # Generate clinical summary
        symptom_log = "\n".join([f"Q: {q['question']}\nA: {q['answer']}" for q in st.session_state.symptom_details])
        patient_profile = f"""
Name: {st.session_state.patient_info['name']}
Age: {st.session_state.patient_info['age']}
Gender: {st.session_state.patient_info['gender']}
Main Symptom: {st.session_state.patient_info['main_symptom']}
Symptom Details:
{symptom_log}
Medical History: {st.session_state.patient_info.get('medical_history', 'N/A')}
Medications: {st.session_state.patient_info.get('medications', 'N/A')}
Allergies: {st.session_state.patient_info.get('allergies', 'N/A')}
        """

        # Risk analysis
        dataset_path = "DATASET.csv"  # Ensure this path is correct
        analysis_system = MedicalAnalysisSystem(dataset_path)
        analysis_results = analysis_system.analyze_patient_data(patient_profile)
        
        # Store the analysis results in session state
        st.session_state.analysis_results = analysis_results
        
        col1 = st.columns(1)
        
        with col1:
            st.subheader("Risk Analysis")
            if "error" in analysis_results:
                st.error(analysis_results["error"])
            else:
                st.metric("Risk Level", analysis_results['risk_label'])
                st.progress(analysis_results['risk_percentage'] / 100)
                st.write(f"*Score*: {analysis_results['risk_score']:.1f}/{analysis_system.MAX_RISK_SCORE:.1f}")
                
        # Download report
        report_content = f"CLINICAL SUMMARY:\n{patient_profile}\n\nRISK ANALYSIS:\n{analysis_results}"
        st.download_button("Download Full Report", report_content, "medical_report.txt")

    except Exception as e:
        st.error(f"Assessment Error: {str(e)}")

def schedule_appointment():
    st.header("🚑 Schedule Specialist Appointment")
    
    # Doctor database
    doctors = [
        {
            'name': 'Dr. Raj Patel',
            'hospital': 'City General Hospital',
            'specialty': 'Cardiology',
            'slots': ['2024-03-25 09:00', '2024-03-25 10:00', '2024-03-26 11:00'],
            'contact': '555-0101',
            'emergency': True
        },
        {
            'name': 'Dr. Siddhart Singh',
            'hospital': 'Metropolitan Health',
            'specialty': 'Neurology',
            'slots': ['2024-03-25 14:00', '2024-03-26 09:30', '2024-03-27 15:00'],
            'contact': '555-0102',
            'emergency': True
        },
        {
            'name': 'Dr. Riya Agarwal',
            'hospital': 'Sunrise Clinic',
            'specialty': 'General Practice',
            'slots': ['2024-03-24 10:00', '2024-03-25 11:00', '2024-03-26 16:00'],
            'contact': '555-0103',
            'emergency': False
        },
        {
            'name': 'Dr. Sarvanan',
            'hospital': 'Westside Medical Center',
            'specialty': 'Orthopedics',
            'slots': ['2024-03-25 08:00', '2024-03-26 10:00', '2024-03-27 09:00'],
            'contact': '555-0104',
            'emergency': True
        },
        {
            'name': 'Dr. Kuldeep Mishra',
            'hospital': "Children's Hospital",
            'specialty': 'Pediatrics',
            'slots': ['2024-03-25 12:00', '2024-03-26 14:00', '2024-03-27 10:00'],
            'contact': '555-0105',
            'emergency': True
        }
    ]
    
    risk_data = st.session_state.get('analysis_results', {})
    
    # Check if risk_data is None or empty
    if not risk_data or "error" in risk_data:
        st.error("No risk assessment available. Please complete the assessment first.")
        return
    
    risk_label = risk_data.get('risk_label', 'Low')
    
    # Priority explanation
    st.markdown(f"""
    <div class="priority-banner">
        Your current risk level: <strong>{risk_label}</strong> priority
        <br>{(risk_label == 'High') and '🟥 Urgent - Same day appointments available' 
            or (risk_label == 'Medium') and '🟨 Semi-Urgent - Next day appointments' 
            or '🟩 Routine - Book within 3 days'}
    </div>
    """, unsafe_allow_html=True)
    
    # Filter doctors based on risk
    if risk_label == 'High':
        available_doctors = [d for d in doctors if d['emergency']]
    else:
        available_doctors = doctors
    
    # Display doctors in columns
    cols = st.columns(2)
    for idx, doctor in enumerate(available_doctors):
        with cols[idx % 2]:
            with st.container():
                st.subheader(f"🏥 {doctor['hospital']}")
                st.markdown(f"""
                *Doctor*: {doctor['name']}  
                *Specialty*: {doctor['specialty']}  
                *Contact*: {doctor['contact']}
                """)
                
                # Sort slots based on risk
                slots = sorted(doctor['slots'], key=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M'))
                if risk_label == 'Low':
                    slots = slots[::-1]
                
                selected_slot = st.selectbox(f"Available slots with {doctor['name']}", 
                                           slots, 
                                           key=f"slot_{idx}")
                
                if st.button(f"Book with {doctor['name']}", key=f"book_{idx}"):
                    st.session_state.appointment_details = {
                        'doctor': doctor['name'],
                        'hospital': doctor['hospital'],
                        'time': selected_slot,
                        'contact': doctor['contact'],
                        'risk_level': risk_label
                    }
                    st.success("Appointment booked successfully!")
                    st.balloons()
                    
                    # Generate appointment summary
                    summary = f"""
                    *Patient Name*: {st.session_state.patient_info['name']}
                    *Age*: {st.session_state.patient_info['age']}
                    *Booked Appointment*:
                    - Doctor: {doctor['name']}
                    - Hospital: {doctor['hospital']}
                    - Time: {selected_slot}
                    - Contact: {doctor['contact']}
                    - Priority Level: {risk_label}
                    """
                    st.session_state.appointment_summary = summary
                    
                    # Show download button
                    st.download_button("Download Appointment Details", 
                                      summary, 
                                      "appointment_confirmation.txt",
                                      help="Save your appointment details")

def main():
    st.title("🏥 AI Medical Consultancy")
    
    # Initialize Groq client
    if not initialize_groq_client():
        st.warning("Please provide a valid Groq API key to proceed.")
        return

    # Define steps for the progress bar
    steps = ["Patient Info", "Symptoms", "History", "Report", "Booking"]
    
    # Display progress bar
    progress = f"""
    <div class="progress-bar">
        {"".join(f'<div class="step {"active" if st.session_state.current_step >= i else ""}">{i+1}. {step}</div>'
        for i, step in enumerate(steps))}
    </div>
    """
    st.markdown(progress, unsafe_allow_html=True)

    # Step routing logic
    if st.session_state.current_step == 0:
        collect_basic_info()  # Step 1: Collect patient information
    elif st.session_state.current_step in [1, 2]:
        handle_symptom_interrogation()  # Step 2: Symptom analysis
    elif st.session_state.current_step == 3:
        collect_medical_history()  # Step 3: Collect medical history
    elif st.session_state.current_step == 4:
        generate_risk_assessment()  # Step 4: Generate risk assessment
        if st.button("📅 Schedule Doctor Appointment"):
            st.session_state.current_step = 5  # Move to the booking step
            st.rerun()
    elif st.session_state.current_step == 5:
        schedule_appointment()  # Step 5: Schedule appointment with a doctor

    # Debugging: Show session state (optional)
    if st.sidebar.checkbox("Show Session State (Debug)"):
        st.sidebar.write(st.session_state)

if __name__ == "__main__":
    main()