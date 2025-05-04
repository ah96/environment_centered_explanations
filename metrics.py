def compute_path_metrics(original_path, new_path, original_time, new_time, grid_size):
    original_length = len(original_path) if original_path else 0
    new_length = len(new_path) if new_path else 0
    path_restored = original_length == 0 and new_length > 0

    if original_length > 0:
        path_length_diff = new_length - original_length
        path_length_change_pct = (path_length_diff / original_length) * 100
    else:
        # Use a finite penalty based on grid size
        penalty = grid_size * grid_size + 1
        path_length_change_pct = 100 * (1 - (new_length / penalty)) if new_length else 0
        path_length_diff = new_length  # Treat this as improvement if original was blocked

    exec_time_diff = new_time - original_time
    path_success = new_length > 0

    return {
        "original_length": original_length,
        "new_length": new_length,
        "path_restored": path_restored,
        "path_length_diff": path_length_diff,
        "path_length_change_pct": path_length_change_pct,
        "exec_time_diff": exec_time_diff,
        "path_success": path_success,
    }
