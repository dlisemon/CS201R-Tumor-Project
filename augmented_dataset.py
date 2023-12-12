import random 
import pandas as pd

# Tumor grade probabilities based on other features  
grade_distribution = {
    (True, True): [0.1, 0.2, 0.3, 0.4],
    (True, False): [0.05, 0.3, 0.5, 0.15], 
    (False, True): [0.4, 0.35, 0.2, 0.05],
    (False, False): [0.8, 0.15, 0.05, 0]  
}

def create_patient(idh1_mut, co_del_19q):
    ki67 = random.randint(0, 100)   
    age = random.randint(30, 80)
    gender = random.choice(['M', 'F'])
    
    # Sample grade based on biomarkers
    grade_probs = grade_distribution[(idh1_mut, co_del_19q)]
    grade = random.choices([1, 2, 3, 4], grade_probs)[0]  
    
    return [age, gender, ki67, grade, idh1_mut, co_del_19q]

# Generate balanced dataset with random correlations       
rows = []
for i in range(1000):
    idh1_mut = random.choice([True, False])
    co_del_19q = random.choice([True, False]) 
    rows.append(create_patient(idh1_mut, co_del_19q))
    
df = pd.DataFrame(rows, columns=['age', 'gender', 'ki67', 'grade', 'idh1', 'del19q' ])

def save_to_csv(df, file_name):
    df.to_csv(f'{file_name}.csv', index=False)
    
save_to_csv(df, 'augmented_data') 
print('Saved augmented data to augmented_data.csv')