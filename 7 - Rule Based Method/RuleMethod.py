def apply_discount(age):
    # Define rules
    if age < 18:
        return "Not eligible for a discount."
    elif 18 <= age < 30:
        return "You qualify for a 10% discount."
    elif 30 <= age < 50:
        return "You qualify for a 20% discount."
    else:
        return "You qualify for a 30% discount."

# Test cases
test_cases = [15, 25, 35, 55]
for age in test_cases:
    result = apply_discount(age)
    print(f"Age: {age} - {result}")
