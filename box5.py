import pinn

if __name__ == "__main__":
    effect_size = 0.5
    alpha = 0.0001
    sample_size = 200
    rho_ij = 0.25
    rho_uv = 0.81

    edge_ij_power = pinn.power.tt_ind_solve_power(
        effect_size, sample_size, alpha, reliability=rho_ij
    )
    edge_uv_power = pinn.power.tt_ind_solve_power(
        effect_size, sample_size, alpha, reliability=rho_uv
    )

    print(
        "Power in edge {ij} with reliability " f"{rho_ij} is {edge_ij_power*100:.2f}%"
    )
    print(
        "Power in edge {uv} with reliability " f"{rho_uv} is {edge_uv_power*100:.2f}%"
    )
