from tools.quantized_optimizers.recording_split import components


def test_trials_and_cameras_share_a_recording_and_identical_media_join_families():
    rows = [
        {"id": "20230728_37_CN38_T1_Camera1_0_0", "path": "/root/data/wts_eval/a.mp4", "sha256": "a" * 64},
        {"id": "20230728_37_CN38_T2_Camera3_1_1", "path": "/root/data/wts_train/b.mp4", "sha256": "b" * 64},
        {"id": "20230929_12_CN38_T1_Camera1_0_0", "path": "/root/data/wts_train/c.mp4", "sha256": "b" * 64},
        {"id": "20230929_13_SN25_T1_Camera1_0_0", "path": "/root/data/wts_train/d.mp4", "sha256": "d" * 64},
    ]
    result = components(rows)
    assert len({result[key]["group"] for key in ("wts_eval/a.mp4", "wts_train/b.mp4", "wts_train/c.mp4")}) == 1
    assert result["wts_train/d.mp4"]["group"] != result["wts_eval/a.mp4"]["group"]


def test_final_selection_ignores_source_order_and_excludes_whole_groups():
    import json

    import pytest

    from tools.quantized_optimizers.select_final import select

    inventory = {f"eval/{i}.mp4": {"group": str(i // 2)} for i in range(8)}
    lines = [
        json.dumps({"id": str(i), "messages": [{"content": [{"type": "video", "video": f"/root/data/eval/{i}.mp4"}]}]})
        + "\n"
        for i in range(8)
    ]
    result = select(lines, inventory, {"0"}, count=3)
    assert result == select(list(reversed(lines)), inventory, {"0"}, count=3)
    assert set(result[1]) == {"1", "2", "3"}
    assert len(result[0]) == 3
    with pytest.raises(ValueError, match="eligible recording groups"):
        select(lines, inventory, {"0"}, count=4)
