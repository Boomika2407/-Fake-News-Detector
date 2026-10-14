import time
import json
import sys
from lime_api_server import app, initialize_models

# A very long mock news article (~10,000 characters)
very_long_text = """
The federal government today announced a major overhaul of the nation's energy infrastructure, pledging over $100 billion in funding for renewable energy projects over the next decade. The initiative, dubbed the "Green Future Act," aims to modernize the power grid, expand wind and solar capacity, and phase out coal-fired power plants. Proponents of the plan argue it will create hundreds of thousands of jobs, reduce carbon emissions, and lower energy costs for consumers. However, critics from the fossil fuel industry and some political opposition members argue the plan is overly ambitious, will lead to energy reliability issues, and represents a massive government overreach that will hurt the economy. 
The package includes subsidies for home solar installations, tax credits for electric vehicle purchases, and funding for research into battery storage technologies. Several major utilities have already expressed cautious support, while warning that upgrading the grid to handle the influx of intermittent renewable energy will be a massive logistical challenge. Environmental groups have hailed the announcement as a historic step forward, though some argue it does not go far enough or fast enough to address the climate crisis.
As the bill heads to Congress, a fierce legislative battle is expected. Senators from energy-producing states have vowed to block or modify key provisions, raising concerns about the impact on local economies that depend on oil, gas, and coal. The administration maintains that the transition is inevitable and that failing to act now will cost the country more in the long run due to the increasing frequency and severity of climate-related disasters.
Meanwhile, international observers are watching closely, as the policy shift could influence global climate negotiations and green technology supply chains. Analysts suggest that the success of the initiative will depend heavily on the rapid deployment of grid-scale storage and the cooperation of private sector partners. Over the coming months, public hearings and committee debates will shape the final details of the legislation, with both sides mobilizing public relations campaigns to win support.
Additionally, the bill addresses environmental justice issues, earmarking a portion of the funds for communities that have historically borne the brunt of industrial pollution. This aspect has garnered praise from civil rights organizations, though implementation details remain vague. The debate highlights the complex intersection of economics, technology, and politics in the transition to a low-carbon future, a challenge that will define the coming decades.
""" * 4  # Repeat to make it ~10k characters

def run_performance_test():
    print("Initializing models and explainer...")
    initialize_models()
    
    print("\nCreating Flask test client...")
    client = app.test_client()
    
    print(f"\nSending prediction request for long text ({len(very_long_text)} characters)...")
    start_time = time.time()
    
    response = client.post('/predict', json={
        'text': very_long_text,
        'threshold': 0.0  # Force LIME explanation regardless of confidence
    })
    
    elapsed = time.time() - start_time
    print(f"Response status code: {response.status_code}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    
    assert response.status_code == 200, f"Expected status 200, got {response.status_code}"
    
    data = json.loads(response.data)
    print("\nPrediction Result:")
    print(f"  Prediction: {data.get('prediction')}")
    print(f"  Confidence: {data.get('confidence'):.4f}")
    
    # Check explanation structure
    explanation = data.get('explanation', {})
    assert 'features' in explanation, "No LIME features in the response explanation!"
    print(f"\nExplanation features found: {len(explanation['features'])}")
    for feat in explanation['features'][:5]:
        print(f"  Word: {feat['feature']}, Weight: {feat['weight']:.4f}, Direction: {feat['direction']}")
        
    print("\nLinguistic Analysis found:")
    linguistic = data.get('linguistic_analysis', {})
    print(f"  Combined insight: {linguistic.get('combined_insight')}")
    
    # Assert speed is within acceptable limits (should be < 15 seconds)
    print(f"\nAssertion: Speed should be under 15 seconds...")
    if elapsed <= 15.0:
        print("✓ SUCCESS: Speed test passed!")
        sys.exit(0)
    else:
        print("❌ FAILURE: Explanation took longer than 15 seconds!")
        sys.exit(1)

if __name__ == '__main__':
    run_performance_test()
