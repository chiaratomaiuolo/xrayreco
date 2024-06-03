from X-ray_hit_reconstruction.preprocessing import 





if __name__ == "__main__":
    input_file_path = '/Users/chiara/hexsampledata/sim_HexagonalLayout.ODD_Rum_20enc_srcsigma200um.h5'
    data = Xraydata(input_file_path)
    print(data.input_events_data()[1])
    print(data.target_data())
    del data