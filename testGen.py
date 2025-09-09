import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate data
samples = 50
sizes = np.random.randint(30, 201, size=samples)
standards =  np.random.uniform(1, 4, size=samples)

# Calculate house prices
prices = [
    int(3000 * (size**0.7) * np.sqrt(standard) * np.random.randint(95,105))
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



