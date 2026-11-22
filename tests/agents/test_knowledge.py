from kernel_study.agents.knowledge import select_cards


def test_knowledge_selection_is_shared_deterministic_and_ablatable():
    a = select_cards('softmax')
    assert a == select_cards('softmax')
    assert any(c['category']=='fusion' for c in a)
    assert all(c['category']!='fusion' for c in select_cards('softmax',excluded_category='fusion'))
    assert {c['id'] for c in a} == {c['id'] for c in select_cards('softmax',{'kernels':[{'spills':8}]})}
