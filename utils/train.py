def train_dual_network(
    model_u,
    model_k,
    X_obs_train,
    U_obs_train,
    X_obs_k_train,
    K_obs_train,
    X_pde_train,
    F_pde_train,
    X_obs_test,
    U_obs_test,
    X_obs_k_test,
    K_obs_test,
    X_pde_test,
    F_pde_test,
    adam_lr=1e-3,
    adam_iters=1000,
    lbfgs_iters=2000,
    verbose=False,
    print_every=100, 
    lambda_pde_scheduler=True,
    adaptive_weights=True,
    alpha=5,
    update_every=200, 
    regularization=False,    
):
    ratio = 1
    criterion = nn.MSELoss()

    parameters = (
        list(model_u.parameters())
        + list(model_k.parameters())
    )

    optimizer_adam = optim.AdamW(
        parameters,
        lr=adam_lr ,
        weight_decay=1 if regularization else 0.0
    )

    optimizer_lbfgs = optim.LBFGS(
        parameters,
        lr=1,
        max_iter=lbfgs_iters,
        max_eval=lbfgs_iters,
        history_size=100,
        tolerance_change=1.0 * np.finfo(float).eps,
        line_search_fn="strong_wolfe")

 

    history = {
        "total": [],
        "u": [],
        "k": [],
        "pde": [],

        "lambda_u": [],
        "lambda_k": [],
        "lambda_pde": [],

        # New entries
        "R_u": [],
        "R_k": [],
        "R_pde": [],
        "R_reg_t": [],
        "ratio": [],
        "total_test": [],
        "total_no_reg": [],
        "total_no_reg_test": [],
    }

    # --------------------------------------------------
    # Adaptive weights
    # --------------------------------------------------

    lambda_u = 1.0
    lambda_k = 1.0
    lambda_pde = 1.0
 
    # --------------------------------------------------
    # Helper: compute losses
    # --------------------------------------------------



    def compute_losses(lambda_reg=1.0):

        loss_u = observation_loss_u(
            model_u,
            X_obs_train,
            U_obs_train,
            criterion,
        )

        loss_k = observation_loss_k(
            model_k,
            X_obs_k_train,
            K_obs_train,
            criterion,
        )

        loss_pde = pde_loss_inf(
            model_u,
            model_k,
            X_pde_train,
            F_pde_train,
        )

        loss_u = torch.nan_to_num(loss_u)
        loss_k = torch.nan_to_num(loss_k)
        loss_pde = torch.nan_to_num(loss_pde)


        if regularization:
            reg_t = l2_regularization(parameters)
            
  
        total = (
            lambda_u * loss_u
            + lambda_k * loss_k
            + lambda_pde * loss_pde
            + (lambda_reg * reg_t if regularization else 0.0)
        )

        total_no_reg = (
             loss_u
            + loss_k
            + loss_pde
        )

        return total, loss_u, loss_k, loss_pde, total_no_reg

    def compute_losses_test(lambda_reg=1.0):

        loss_u = observation_loss_u(
            model_u,
            X_obs_test,
            U_obs_test,
            criterion,
        )

        loss_k = observation_loss_k(
            model_k,
            X_obs_k_test,
            K_obs_test,
            criterion,
        )

        loss_pde = pde_loss_inf(
            model_u,
            model_k,
            X_pde_test,
            F_pde_test,
        )

        loss_u = torch.nan_to_num(loss_u)
        loss_k = torch.nan_to_num(loss_k)
        loss_pde = torch.nan_to_num(loss_pde)

        if regularization:
            reg_t = l2_regularization(parameters)
 
        total = (
            lambda_u * loss_u
            + lambda_k * loss_k
            + lambda_pde * loss_pde
            + (lambda_reg * reg_t if regularization else 0.0)
        )

        total_no_reg = (
            loss_u
            + loss_k
            + loss_pde
        )

        return total, loss_u, loss_k, loss_pde, total_no_reg

 
    def ratio_calculation():
        V = np.array([
            np.mean(history["u"][-update_every:]),
            np.mean(history["k"][-update_every:]),
            np.mean(history["pde"][-update_every:]),
            #np.mean(history["reg_t"][-update_every:])
        ])

        # ----------------------------------------------------
        # Step 7: Ratio
        # ----------------------------------------------------

        ratio = V.max() / (V.min() + 1e-12)
        return ratio


    def get_pde_weight(epoch):

        if epoch < 250:
            return 0.10

        elif epoch < 500:
            return 0.25

        elif epoch < 750:
            return 0.50

        elif epoch < 1000:
            return 0.75

        else:
            return 1.00


    def update_loss_weights(loss_u, loss_k):

        nonlocal lambda_u, lambda_k

        # ----------------------------------------------------
        # Current U and K losses
        # ----------------------------------------------------

        if len(history["u"]) == 0:

            V = np.array([
                loss_u.item(),
                loss_k.item(),
            ])

        else:

            V = np.array([
                np.mean(history["u"][-update_every:]),
                np.mean(history["k"][-update_every:]),
            ])

        # ----------------------------------------------------
        # Ratio between U and K
        # ----------------------------------------------------

        ratio = V.max() / (V.min() + 1e-12)

        if not adaptive_weights:
            return

        ratio_threshold = 10.0

        if ratio <= ratio_threshold:

            history["R_u"].append(0.0)
            history["R_k"].append(0.0)

            return

        # ----------------------------------------------------
        # Compute relative difference
        # ----------------------------------------------------

        R = (V - V.min()) / (
            V.max() - V.min() + 1e-12
        )

        history["R_u"].append(R[0])
        history["R_k"].append(R[1])

        # ----------------------------------------------------
        # Update only U and K
        # ----------------------------------------------------

        lambdas = 1.0 + alpha * R

        # Loss with smaller magnitude keeps weight = 1
        fastest = np.argmin(V)
        lambdas[fastest] = 1.0

        lambda_u = lambdas[0]
        lambda_k = lambdas[1]

        if verbose:
            print(
                f"V      = {V.round(3)}\n"
                f"R      = {R.round(3)}\n"
                f"ratio  = {ratio:.2f}\n"
                f"lambda_u = {lambda_u:.3f}\n"
                f"lambda_k = {lambda_k:.3f}\n"
                f"lambda_pde = {lambda_pde:.3f}"
            )
 
        
    # --------------------------------------------------
    # Helper: save history
    # --------------------------------------------------

    def save_history(total, loss_u, loss_k, loss_pde, ratio, total_test, total_no_reg,total_no_reg_test):

        history["total"].append(total.item())
        history["u"].append(loss_u.item())
        history["k"].append(loss_k.item())
        history["pde"].append(loss_pde.item())
        #history["reg_t"].append(reg_t.item())
        history["lambda_u"].append(lambda_u)
        history["lambda_k"].append(lambda_k)
        history["lambda_pde"].append(lambda_pde)
        #history["lambda_reg_t"].append(lambda_reg_t)
        history["ratio"].append(ratio)
        history["total_test"].append(total_test.item())
        history["total_no_reg"].append(total_no_reg.item())
        history["total_no_reg_test"].append(total_no_reg_test.item())

    # --------------------------------------------------
    # Adam
    # --------------------------------------------------
    if verbose:
        print("\n====================================")
        print("Training with Adam")
        print("====================================")

    model_u.train()
    model_k.train()

    for epoch in range(adam_iters+1):

        if lambda_pde_scheduler is True:
            lambda_pde = get_pde_weight(epoch)

        optimizer_adam.zero_grad()

        # total_no_reg_test = torch.tensor(0.0)  # Initialize total_no_reg_test to avoid undefined variable error
        # total_test = torch.tensor(0.0)  # Initialize total_test to avoid undefined
        #lambda_reg=0

        total, loss_u, loss_k, loss_pde, total_no_reg = compute_losses()
        total_test, loss_u_test, loss_k_test, loss_pde_test, total_no_reg_test = compute_losses_test()

        if epoch == 0:

            V = np.array([
                loss_u.item(),
                loss_k.item(),
                loss_pde.item(),
                #reg_t.item()
            ])

            ratio = V.max() / (V.min() + 1e-12)
        #     update_loss_weights(loss_u, loss_k, loss_pde) 
        #     total, loss_u, loss_k, loss_pde, total_no_reg = compute_losses()
        #     total_test, loss_u_test, loss_k_test, loss_pde_test, total_no_reg_test = compute_losses_test()

        else:
            ratio = ratio_calculation()
        

        total.backward()
        optimizer_adam.step()

        #expect in zero 


        if adaptive_weights and epoch % update_every == 0 and epoch > 0:
            update_loss_weights(loss_u, loss_k)
 


        if epoch % print_every == 0:
            save_history(total, loss_u, loss_k, loss_pde, ratio, total_test, total_no_reg, total_no_reg_test)

        if verbose and epoch % print_every == 0:
            print(
                f"Adam {epoch:5d} | "
                f"Total={total.item():.3e} | "
                f"ObsU={loss_u.item():.3e} | "
                f"ObsK={loss_k.item():.3e} | "
                f"PDE={loss_pde.item():.3e} | "
                f"Ratio={ratio:.2f}"
            )


    # --------------------------------------------------
    # L-BFGS
    # --------------------------------------------------

    if verbose:
        print("\n====================================")
        print("Training with L-BFGS")
        print("====================================")

    state = {
        "iter": 0,
        "loss_u": None,
        "loss_k": None,
        "loss_pde": None,
    }

    iters_done = 0
    #while iters_done < lbfgs_iters-1:

        # --------------------------------------------------
        # Restart L-BFGS every update_every iterations
        # --------------------------------------------------

        #current_block = min(update_every, lbfgs_iters - iters_done)

        # optimizer_lbfgs = torch.optim.LBFGS(
        #     list(model_u.parameters()) + list(model_k.parameters()),
        #     lr=1.0,
        #     max_iter=current_block,
        #     max_eval=current_block,
        #     history_size=100,
        #     tolerance_grad=1e-9,
        #     tolerance_change=1e-12,
        #     line_search_fn="strong_wolfe",
        # )

    def closure():

        optimizer_lbfgs.zero_grad()

        lambda_reg = 0#1e-3

        total, loss_u, loss_k, loss_pde, total_no_reg = \
            compute_losses(lambda_reg)

        total_test, loss_u_test, loss_k_test, loss_pde_test, total_no_reg_test = \
            compute_losses_test(lambda_reg)

        total.backward()

        # ------------------------------------------
        # Ratio
        # ------------------------------------------

        if iters_done == 0:

            V = np.array([
                loss_u.item(),
                loss_k.item(),
                loss_pde.item(),
            ])

            ratio = V.max() / (V.min() + 1e-12)

        else:

            ratio = ratio_calculation()

        # ------------------------------------------
        # Store latest losses
        # ------------------------------------------

        state["loss_u"] = loss_u.detach()
        state["loss_k"] = loss_k.detach()
        state["loss_pde"] = loss_pde.detach()

        # ------------------------------------------
        # Save history
        # ------------------------------------------

        if state["iter"] % print_every == 0:
            save_history(
                total,
                loss_u,
                loss_k,
                loss_pde,
                ratio,
                total_test,
                total_no_reg,
                total_no_reg_test,
            )

        state["iter"] += 1

        if verbose and state["iter"] % print_every == 0:

            print(
                f"L-BFGS {state['iter']:5d} | "
                f"Total={total.item():.3e} | "
                f"ObsU={loss_u.item():.3e} | "
                f"ObsK={loss_k.item():.3e} | "
                f"PDE={loss_pde.item():.3e} | "
                f"Ratio={ratio:.2f}"
            )

        return total

    # --------------------------------------------------
    # Run one L-BFGS block
    # --------------------------------------------------

    optimizer_lbfgs.step(closure)
    #iters_done += current_block
    # --------------------------------------------------
    # Update adaptive weights
    # --------------------------------------------------

    # if adaptive_weights and iters_done < lbfgs_iters:

    #     update_loss_weights(
    #         state["loss_u"],
    #         state["loss_k"],
    #         state["loss_pde"],
    #     )



    # --------------------------------------------------
    # Save history automatically
    # --------------------------------------------------

    os.makedirs("results", exist_ok=True)

    filename = "history"

    if adaptive_weights:
        filename += "_adaptive"
    else:
        filename += "_fixed"

    if regularization:
        filename += "_reg"
    else:
        filename += "_no_reg"

    filename += ".pkl"

    with open(os.path.join("results", filename), "wb") as f:
        pickle.dump(history, f)

    #print(f"History saved to results/{filename}")

    return history