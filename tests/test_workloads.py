import pytest

from src.workloads.length_batch import LengthBatch
from src.workloads.uniform import make_uniform_workload


def test_length_batch_computes_summary_fields():
    batch = LengthBatch(
        prompt_lengths=[4, 8, 16],
        decode_lengths=[2, 2, 4],
    )

    assert batch.batch_size == 3
    assert batch.total_lengths == [6, 10, 20]
    assert batch.seq_len_min == 6
    assert batch.seq_len_max == 20
    assert batch.seq_len_mean == pytest.approx(12.0)


def test_length_batch_rejects_empty_prompt_lengths():
    with pytest.raises(ValueError):
        LengthBatch(prompt_lengths=[], decode_lengths=[])


def test_length_batch_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        LengthBatch(prompt_lengths=[4, 8], decode_lengths=[2])


def test_length_batch_rejects_nonpositive_prompt_lengths():
    with pytest.raises(ValueError):
        LengthBatch(prompt_lengths=[4, 0], decode_lengths=[1, 1])


def test_length_batch_rejects_negative_decode_lengths():
    with pytest.raises(ValueError):
        LengthBatch(prompt_lengths=[4, 8], decode_lengths=[1, -1])


def test_make_uniform_workload():
    workload = make_uniform_workload(
        batch_size=4,
        prompt_len=128,
        decode_len=60,
    )

    assert workload.batch_size == 4
    assert workload.prompt_lengths == [128, 128, 128, 128]
    assert workload.decode_lengths == [60, 60, 60, 60]
    assert workload.total_lengths == [188, 188, 188, 188]
    assert workload.seq_len_min == 188
    assert workload.seq_len_mean == pytest.approx(188.0)
    assert workload.seq_len_max == 188


@pytest.mark.parametrize(
    "batch_size,prompt_len,decode_len",
    [
        (0, 128, 60),
        (4, 0, 60),
        (4, 128, -1),
    ],
)
def test_make_uniform_workload_rejects_invalid_inputs(
    batch_size,
    prompt_len,
    decode_len,
):
    with pytest.raises(ValueError):
        make_uniform_workload(
            batch_size=batch_size,
            prompt_len=prompt_len,
            decode_len=decode_len,
        )
