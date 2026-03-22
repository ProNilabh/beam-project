import httpx
import json
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    print("1. Testing GET / (health check) ...")
    r = httpx.get(f"{BASE_URL}/")
    assert r.status_code == 200
    data = r.json()
    print(f"    Status: {data['status']} | Model: {data['model']}")
    return True


def test_model_info():
    print("2. Testing GET /model-info ...")
    r = httpx.get(f"{BASE_URL}/model-info")
    assert r.status_code == 200
    data = r.json()
    print(f"    Model: {data['model_name']} | R²: {data['test_r2']}")
    return True


def test_single_predict():
    print("3. Testing POST /predict (single building) ...")
    payload = {
        "Relative_Compactness": 0.74,
        "Surface_Area": 686.0,
        "Wall_Area": 245.0,
        "Roof_Area": 220.5,
        "Overall_Height": 3.5,
        "Orientation": 3.0,
        "Glazing_Area": 0.1,
        "Glazing_Area_Distribution": 2.0,
    }
    r = httpx.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    print(f"    Heating: {data['Heating_Load_kWh_m2']} kWh/m²")
    print(f"    Cooling: {data['Cooling_Load_kWh_m2']} kWh/m²")
    return True


def test_batch_predict():
    print("4. Testing POST /predict/batch (2 buildings) ...")
    payload = {
        "buildings": [
            {
                "Relative_Compactness": 0.98,
                "Surface_Area": 514.5,
                "Wall_Area": 294.0,
                "Roof_Area": 110.25,
                "Overall_Height": 7.0,
                "Orientation": 2.0,
                "Glazing_Area": 0.0,
                "Glazing_Area_Distribution": 0.0,
            },
            {
                "Relative_Compactness": 0.62,
                "Surface_Area": 808.5,
                "Wall_Area": 367.5,
                "Roof_Area": 220.5,
                "Overall_Height": 3.5,
                "Orientation": 4.0,
                "Glazing_Area": 0.4,
                "Glazing_Area_Distribution": 5.0,
            },
        ]
    }
    r = httpx.post(f"{BASE_URL}/predict/batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    print(f"    Returned {data['count']} predictions")
    for i, p in enumerate(data["predictions"]):
        print(f"      Building {i+1}: Heating={p['Heating_Load_kWh_m2']}, Cooling={p['Cooling_Load_kWh_m2']}")
    return True


def test_swagger():
    print("5. Testing GET /docs (Swagger UI) ...")
    r = httpx.get(f"{BASE_URL}/docs")
    assert r.status_code == 200
    print("    Swagger UI is accessible at http://localhost:8000/docs")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  BEAM API TEST SUITE")
    print("=" * 60)
    print()

    tests = [test_health, test_model_info, test_single_predict, test_batch_predict, test_swagger]
    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except httpx.ConnectError:
            print("    Connection refused — is the server running?")
            print("      Start it with: uvicorn src.app:app --reload")
            sys.exit(1)
        except AssertionError as e:
            print(f"    FAILED: {e}")
        print()

    print("=" * 60)
    print(f"  Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)
