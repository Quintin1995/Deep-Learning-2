# Splits up the Q-values for a state for all actions into estimated value and advantages.
# Returns list of tuples (value, advantage)
def value_and_advantage (q_values):

    # Calculate the average Q-value
    v = sum(q_values) / len(q_values)

    # Subtract the average V from the values to calculate advantage
    return [(v, q - v) for q in q_values]