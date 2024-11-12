from django.views.decorators.csrf import csrf_exempt
import numpy as np
import joblib
from django.http import JsonResponse
from statistics import mode
import os
import json
print(os.getcwd())  # This will print the current working directory

# Load the models once at the start
rf_model = joblib.load('models/rf_model.joblib')
nb_model = joblib.load('models/nb_model.joblib')
svm_model = joblib.load('models/svm_model.joblib')

result = [
    "Paromysal Positional Vertigo (Vertigo)",
    "Acne",
    "AIDS",
    "Alcoholic Hepatitis",
    "Allergy",
    "Arthritis",
    "Bronchial Asthma",
    "Cervical spondylosis",
    "Chicken Pox",
    "Common Cold",
    "Dengue",
    "Diabetes",
    "Dimorphic hemmorhoids (Piles)",
    "Drug Reaction",
    "Fungal Infection",
    "Gastroenteritis",
    "GERD",
    "Heart Attack",
    "Hepatitis A",
    "Hepatitis B",
    "Hepatitis C",
    "Hepatitis D",
    "Hepatitis E",
    "Hypertension",
    "Hypoglycemia",
    "Hypothyroidism",
    "Impetigo",
    "Jaundice",
    "Malaria",
    "Migraine",
    "Osteoarthritis",
    "Paralysis (brain hemorrhage)",
    "Peptic Ulcer Disease",
    "Pneumonia",
    "Psoriasis",
    "Tuberculosis",
    "Typhoid",
    "Urinary Tract Infection",
    "Varicose Veins"
]

# Define the list of columns in the same order as in the dataset
columns = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", 
    "shivering", "chills", "joint_pain", "stomach_pain", "acidity", 
    "ulcers_on_tongue", "muscle_wasting", "vomiting", "burning_micturition", 
    "spotting_urination", "fatigue", "weight_gain", "anxiety", 
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", 
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", 
    "high_fever", "sunken_eyes", "breathlessness", "sweating", "dehydration", 
    "indigestion", "headache", "yellowish_skin", "dark_urine", "nausea", 
    "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "constipation", 
    "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", 
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", 
    "swelling_of_stomach", "swelled_lymph_nodes", "malaise", 
    "blurred_and_distorted_vision", "phlegm", "throat_irritation", 
    "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", 
    "chest_pain", "weakness_in_limbs", "fast_heart_rate", 
    "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", 
    "irritation_in_anus", "neck_pain", "dizziness", "cramps", "bruising", 
    "obesity", "swollen_legs", "swollen_blood_vessels", "puffy_face_and_eyes", 
    "enlarged_thyroid", "brittle_nails", "swollen_extremeties", 
    "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips", 
    "slurred_speech", "knee_pain", "hip_joint_pain", "muscle_weakness", 
    "stiff_neck", "swelling_joints", "movement_stiffness", "spinning_movements", 
    "loss_of_balance", "unsteadiness", "weakness_of_one_body_side", 
    "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine", 
    "continuous_feel_of_urine", "passage_of_gases", "internal_itching", 
    "toxic_look_(typhos)", "depression", "irritability", "muscle_pain", 
    "altered_sensorium", "red_spots_over_body", "belly_pain", 
    "abnormal_menstruation", "dischromic_patches", "watering_from_eyes", 
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum", 
    "rusty_sputum", "lack_of_concentration", "visual_disturbances", 
    "receiving_blood_transfusion", "receiving_unsterile_injections", "coma", 
    "stomach_bleeding", "distention_of_abdomen", "history_of_alcohol_consumption", 
    "fluid_overload", "blood_in_sputum", "prominent_veins_on_calf", 
    "palpitations", "painful_walking", "pus_filled_pimples", "blackheads", 
    "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", 
    "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
]

@csrf_exempt
def predict_disease(request):
    try:
        # Parse the request body as JSON
        data = json.loads(request.body)
        binary_string = data.get("symptoms")
        print(binary_string)

        # Check if symptoms are provided
        if not binary_string:
            return JsonResponse({"error": "No symptoms provided"}, status=400)

        # Convert the binary string to a list of integers
        input_data = list(map(int, binary_string))
        
        # Validate input length
        if len(input_data) != len(columns):
            return JsonResponse({"error": "Invalid symptom input length"}, status=400)

        # Reshape input data for model predictions
        input_data = np.array(input_data).reshape(1, -1)

        # Get predictions from each model
        rf_prediction = rf_model.predict(input_data)[0]
        nb_prediction = nb_model.predict(input_data)[0]
        svm_prediction = svm_model.predict(input_data)[0]

        # Use the mode of predictions for the final diagnosis
        final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])
        print(final_prediction)
        # Convert final prediction to a standard Python int
        final_prediction = int(final_prediction)

        # Send the final diagnosis as a response
        return JsonResponse({"diagnosis": result[final_prediction-1]})

    except json.JSONDecodeError:
        # Handle JSON parsing errors
        return JsonResponse({"error": "Invalid JSON format"}, status=400)
    except Exception as e:
        print(f"Error in predict_disease: {e}")
        return JsonResponse({"error": "Internal Server Error"}, status=500)