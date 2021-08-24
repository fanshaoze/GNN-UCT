good_eff = lambda eff, v_out: eff >= 90
bad_eff = lambda eff, v_out: eff <= 10

good_v = lambda eff, v_out: 15 <= v_out <= 35
bad_v = lambda eff, v_out: 0 <= v_out <= 10

high_reward = lambda eff, v_out: 15 <= v_out <= 35 and eff >= 75
low_reward = lambda eff, v_out: 15 <= v_out <= 35 and eff < 75