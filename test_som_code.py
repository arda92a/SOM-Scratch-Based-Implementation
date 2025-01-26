from som_scratch import TestSOM

if __name__ == "__main__":
    print("===================== Test Start =====================")

    test_instance = TestSOM()

    # Run tests
    print("\n--- Constructor Tests ---")
    test_instance.test_constructor()

    print("\n--- Random Weights Initialization Tests ---")
    test_instance.test_random_weights_init()

    print("\n--- Activation Distance Tests ---")
    test_instance.test_activation_distance()

    print("\n--- Winner Tests ---")
    test_instance.test_winner()

    print("\n--- Win Map Tests ---")
    test_instance.test_win_map()

    print("\n--- Train Tests ---")
    test_instance.test_train()

    print("\n--- Update Weights Tests ---")
    test_instance.test_update_weights()

    print("\n--- Distance Map Tests ---")
    test_instance.test_distance_map()

    print("\n===================== All Tests Completed =====================")
