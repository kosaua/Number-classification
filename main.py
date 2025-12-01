import sys
from workflow import (
    prepare_data_stage,
    quick_prototype_stage,
    optimize_parameters_stage,
    cross_validation_stage,
    train_final_model_stage,
    evaluate_system_stage
)

def print_menu():
    print("\n" + "="*40)
    print("   SYSTEM ROZPOZNAWANIA CYFR (GMM)")
    print("="*40)
    print("1. Przygotowanie danych (MFCC)")
    print("2. Eksperymenty:")
    print("   a) Szybki prototyp")
    print("   b) Optymalizacja parametrów")
    print("   c) Walidacja krzyżowa")
    print("3. Trening modelu końcowego")
    print("4. Ewaluacja systemu")
    print("5. Wyjście")
    print("-" * 40)

def main():
    # Menu Action Mapping
    actions = {
        "1": prepare_data_stage,
        "2a": quick_prototype_stage,
        "2b": optimize_parameters_stage,
        "2c": cross_validation_stage,
        "3": train_final_model_stage,
        "4": evaluate_system_stage,
        "5": lambda: sys.exit(0)
    }

    while True:
        print_menu()
        choice = input("Twój wybór: ").strip().lower()

        action = actions.get(choice)
        
        if action:
            try:
                action()
            except KeyboardInterrupt:
                print("\n\nPrzerwano operację przez użytkownika.")
            except Exception as e:
                print(f"\n[BŁĄD KRYTYCZNY] {e}")
                # Optional: print traceback for debugging
                # import traceback; traceback.print_exc()
            
            input("\nNaciśnij ENTER, aby kontynuować...")
        else:
            print("\n! Nieprawidłowy wybór. Spróbuj ponownie.")

if __name__ == "__main__":
    main()