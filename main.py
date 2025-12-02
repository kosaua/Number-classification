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
    """Displays the main menu of the application."""
    # Using a single multi-line string for cleaner menu presentation.
    menu_text = """
========================================
   SYSTEM ROZPOZNAWANIA CYFR (GMM)
========================================
1. Przygotowanie danych (MFCC)
2. Eksperymenty:
   a) Szybki prototyp
   b) Optymalizacja parametrów
   c) Walidacja krzyżowa
3. Trening modelu końcowego
4. Ewaluacja systemu
5. Wyjście
----------------------------------------
"""
    print(menu_text)


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
                            
            input("\nNaciśnij ENTER, aby kontynuować...")
        else:
            print("\n! Nieprawidłowy wybór. Spróbuj ponownie.")

if __name__ == "__main__":
    main()