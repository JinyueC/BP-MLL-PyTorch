def get_threshold_parameter(train_data, train_target):
    # input: train_data [num_training:M1, in_feature:N]
    # input: train_target [num_training:M1, out_feature:Q]
    num_class = train_target.size()[1]
    num_training = train_target.size()[0]
    # print(train_target.shape)
    Label = []
    not_Label = []
    for i in range(num_training):
        temp_label = []
        temp_not_label = []
        for j in range(num_class):
            if train_target[i][j] == 1:
                temp_label.append(j)
            else:
                temp_not_label.append(j)
        Label.append(temp_label)
        not_Label.append(temp_not_label)
    # print(Label, '\n', not_Label)

    # Left is train_output [num_training:M1, out_feature:Q]
    Left = net(train_data.float()).detach().numpy()
    Right = np.zeros((num_training, 1))
    # print(Left.shape, Right.shape)
    for i in range(num_training):
        temp = Left[i]
        index = np.argsort(temp) # sort value from small to large, index correspondence
        temp = temp[index] # sort value according to index
        candidate = np.zeros(num_class+1)
        candidate[0] = temp[0]-0.1
        for j in range(num_class-1):
            candidate[j+1] = (temp[j]+temp[j+1])/2.0
        candidate[num_class] = temp[num_class-1]+0.1
        miss_class = np.zeros(num_class+1)
        # print(candidate) 
        for j in range(num_class+1):
            temp_notlabels = index[0:j]
            temp_labels = index[j:num_class]
            # check out how many in temp_notlabels but not in not_Label
            # check out how many in temp_labels but not in Label
            false_neg = len(set(temp_notlabels).difference(set(not_Label[i])))
            false_pos = len(set(temp_labels).difference(set(Label[i])))
            # if i==0 :
            #     print("false neg is: ", false_neg, "false pos is: ", false_pos)
            miss_class[j] = false_neg + false_pos
        temp_index = np.argmin(miss_class)
        Right[i][0] = candidate[temp_index]
    # print(Right) # num_training,1

    Left = np.column_stack((Left, np.ones((num_training,1))))
    tempvalue, residuals, rank, s = np.linalg.lstsq(Right,Left,rcond=None)
    # print(x) # slight different from MATLAB when num_training <= num_class
    # Weights_sizepre = tempvalue.reshape(-1)[0:num_class]
    # Bias_sizepre = tempvalue.reshape(-1)[num_class]
    # print(Weights_sizepre, '\n', Bias_sizepre)
    threshold_param = tempvalue.reshape(-1,1)
    # print(threshold_param)
    return torch.from_numpy(threshold_param)
