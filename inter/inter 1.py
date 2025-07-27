def calculate_depth(T, g=9.81, v_sound=343, tol=1e-6, max_iter=1000):
    """Вычисляет глубину высоту по общему времени падения T.
    g нормальное 9.81 v_sound нормальное 343
    при температуре 0 скорость звука = 331"""
    def equation(h):
        return (2 * h / g)**0.5 + h / v_sound - T

    low, high = 0, T * v_sound
    for _ in range(max_iter):
        mid = (low + high) / 2
        f_mid = equation(mid)
        if abs(f_mid) < tol: 
            return mid
        elif f_mid > 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2
print("глубина: "+str(round(calculate_depth(2.8), 2))+" м.")
