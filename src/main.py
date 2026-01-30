from .data_preprocessing import build_processed_movies
from .models.user_based_cf import generate_user_based_recommendations
from .models.svd_model import generate_svd_recommendations
from .models.content_based import generate_content_based_recommendations
from .evaluation import run_evaluation
from .cold_start import generate_cold_start_recommendations
from .models.item_based_cf import generate_item_based_recommendations


def main():
    build_processed_movies()
    print("Processed data created.")

    user_cf_path = generate_user_based_recommendations(target_user_id=1, k=10)
    print(f"User-based CF recommendations written to: {user_cf_path}")

    svd_path = generate_svd_recommendations(target_user_id=1, k=10)
    print(f"SVD recommendations written to: {svd_path}")

    cb_path = generate_content_based_recommendations(target_user_id=1, k=10)
    print(f"Content-based recommendations written to: {cb_path}")
    
    item_cf_path = generate_item_based_recommendations(target_user_id=1, k=10)
    print(f"Item-based CF recommendations written to: {item_cf_path}")


    eval_path = run_evaluation()
    print(f"Evaluation metrics written to: {eval_path}")

    cold_start_path = generate_cold_start_recommendations(k=10)
    print(f"Cold-start recommendations written to: {cold_start_path}")


if __name__ == "__main__":
    main()
