import cntk as C
import cntkx as Cx
import numpy as np


def test_focal_loss():
    ce = C.cross_entropy_with_softmax([[1., 2., 3., 4.]], [[0.35, 0.15, 0.05, 0.45]]).eval()
    fl = Cx.focal_loss_with_softmax([[1., 2., 3., 4.]], [[0.35, 0.15, 0.05, 0.45]], alpha=1, gamma=0).eval()

    np.testing.assert_almost_equal(ce, fl, decimal=6)

    ce = C.cross_entropy_with_softmax([[0, 0, 0.8, 0.2]], [[0, 0, 1, 0]]).eval()
    fl = Cx.focal_loss_with_softmax([[0, 0, 0.8, 0.2]], [[0, 0, 1, 0]], gamma=2).eval()

    np.testing.assert_array_less(fl, ce)
    np.testing.assert_almost_equal(fl, np.array([[0.31306446]], dtype=np.float32), decimal=6)

    ce = C.cross_entropy_with_softmax([[0, 0, 0.2, 0.8]], [[0, 0, 1, 0]]).eval()
    fl = Cx.focal_loss_with_softmax([[0, 0, 0.2, 0.8]], [[0, 0, 1, 0]]).eval()

    np.testing.assert_array_less(fl, ce)

    ce = C.cross_entropy_with_softmax([[0, 0, -0.2, 50]], [[0, 0, 1, 0]]).eval()
    fl = Cx.focal_loss_with_softmax([[0, 0, -0.2, 50]], [[0, 0, 1, 0]]).eval()

    np.testing.assert_equal(ce, fl)


def test_focal_loss_image():
    output = C.input_variable((3, 1, 2))
    target = C.input_variable((3, 1, 2))

    o = np.random.random((3, 1, 2)).astype(np.float32)
    t = np.array([[[0, 1]], [[0, 0]], [[1, 0]]], dtype=np.float32)

    ce = C.cross_entropy_with_softmax(output, target, axis=0).eval({output: o, target: t})
    fl = Cx.focal_loss_with_softmax(output, target, alpha=1, gamma=0, axis=0).eval({output: o, target: t})

    np.testing.assert_almost_equal(ce, fl, decimal=5)


def test_binary_focal_loss():
    output = C.input_variable(1)
    target = C.input_variable(1)

    o = np.array([[0.5]], dtype=np.float32)
    t = np.array([[1.]], dtype=np.float32)

    bce = C.binary_cross_entropy(output, target).eval({output: o, target: t})
    bfl = Cx.binary_focal_loss(output, target, alpha=1, gamma=0).eval({output: o, target: t})

    np.testing.assert_almost_equal(bce, bfl, decimal=5)

    bce = C.binary_cross_entropy(output, target).eval({output: o, target: t})
    bfl = Cx.binary_focal_loss(output, target, alpha=1, gamma=2).eval({output: o, target: t})

    np.testing.assert_array_less(bfl, bce)

    o = np.array([[0.00001]], dtype=np.float32)
    t = np.array([[1.]], dtype=np.float32)

    bce = C.binary_cross_entropy(output, target).eval({output: o, target: t})
    bfl = Cx.binary_focal_loss(output, target, alpha=1, gamma=2).eval({output: o, target: t})
    bfl0 = Cx.binary_focal_loss(output, target, alpha=1, gamma=0).eval({output: o, target: t})

    np.testing.assert_almost_equal(bfl, bce, decimal=0)
    np.testing.assert_almost_equal(bfl0, bfl, decimal=0)


def test_binary_focal_loss_image():
    output = C.input_variable((5, 5))
    target = C.input_variable((5, 5))

    o = np.random.random((1, 5, 5)).astype(np.float32)
    t = (np.random.random((1, 5, 5)) < 0).astype(np.float32)

    bce = C.binary_cross_entropy(output, target).eval({output: o, target: t})
    bfl = Cx.binary_focal_loss(output, target, alpha=1, gamma=0).eval({output: o, target: t})

    np.testing.assert_almost_equal(bce, bfl, decimal=3)

    bce = C.binary_cross_entropy(output, target).eval({output: o, target: t})
    bfl = Cx.binary_focal_loss(output, target, alpha=1, gamma=2).eval({output: o, target: t})

    np.testing.assert_array_less(bfl, bce)

    o = np.random.random((1, 5, 5)).astype(np.float32)
    t = np.zeros((1, 5, 5)).astype(np.float32)

    bce = C.binary_cross_entropy(output, target).eval({output: o, target: t})
    bfl = Cx.binary_focal_loss(output, target, alpha=1, gamma=0).eval({output: o, target: t})

    np.testing.assert_almost_equal(bce, bfl, decimal=2)


def test_adaptive_robust_baron_loss():
    alphas = [2, 1.5, 1, 0.5, 0, -0.5, -2, -1e10]
    for alpha in alphas:
        scale = 1
        a = C.input_variable(10)
        b = C.input_variable(10)

        c = Cx.generalised_robust_barron_loss(a, b, alpha, scale)

        assert c.shape == (10, )

        n1 = np.random.random((1, 10)).astype(np.float32)
        n2 = np.random.random((1, 10)).astype(np.float32)
        c.eval({a: n1, b: n2})

        a = C.input_variable((5, 10))
        b = C.input_variable(10)

        c = Cx.generalised_robust_barron_loss(a, b, alpha, scale)

        assert c.shape == (5, 10,)

        n1 = np.random.random((1, 5, 10)).astype(np.float32)
        n2 = np.random.random((1, 10)).astype(np.float32)
        c.eval({a: n1, b: n2})
