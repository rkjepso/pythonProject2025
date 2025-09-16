import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate data
samples = 100
sizes = np.random.randint(5, 10, size=samples) * np.random.randint(4, 25, size=samples)
standards =  np.round(np.random.uniform(1, 4, size=samples),1)

# Calculate house prices
prices = [
    np.round(0.002 * (size**0.7) * np.sqrt(standard) * np.random.randint(95,105), 1)
    for size, standard in zip(sizes, standards)
]

# Create DataFrame
df = pd.DataFrame({
    'Price': np.round(prices, 2),
    'm2': sizes,
    'Standard': standards
})

# Save to CSV
df.to_csv('house_data.csv', index=False)
print("CSV file 'house_data.csv' created successfully.")



