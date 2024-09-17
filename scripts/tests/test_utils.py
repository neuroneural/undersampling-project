from utils.usp_utils import * 
import pytest
from scipy.io import loadmat
import pandas as pd


def test_scale_noise_zeros_both():
    assert np.array_equal(scale_noise(np.zeros((53, 1200)), np.zeros((53, 1200)), 0), np.zeros((53, 1200)))


def test_scale_noise_ones_noise_zeros_signal():
    assert np.array_equal(scale_noise(np.ones((53, 1200)), np.zeros((53, 1200)), 0), np.zeros((53, 1200)))

def test_scale_noise_ones_signal_zeros_noise():
    assert np.array_equal(scale_noise(np.zeros((53, 1200)), np.ones((53, 1200)), 0), np.zeros((53, 1200)))

def test_scale_noise_all_ones():
    value = 1/(np.sqrt(10))#0.31622776601683794
    array = np.full((53, 1200), value)
    assert np.array_equal(scale_noise(np.ones((53, 1200)), np.ones((53, 1200)), 1 ) , array)

def test_scale_noise_all_ones_snr_zero():
    assert np.array_equal(scale_noise(np.ones((53, 1200)), np.ones((53, 1200)), 0 ) , np.ones((53, 1200)))


def test_scale_noise_positive_snr():
    n = np.ones(53)
    x = 2 * np.ones(53)
    SNR = 3  # Example positive SNR
    result = scale_noise(n, x, SNR)
    
    # The expected scale factor for n is computed based on the formula
    xTx = np.sum(np.square(x))
    nTn = np.sum(np.square(n))
    expected_scale_factor = ((xTx / nTn)**0.5) / (10**(SNR / 2))
    expected_result = expected_scale_factor * n
    
    assert np.allclose(result, expected_result), "Unexpected scaled noise for positive SNR"


def test_scale_noise_negative_snr():
    n = np.ones(53)
    x = 2 * np.ones(53)
    SNR = -3  # Example negative SNR
    result = scale_noise(n, x, SNR)
    
    # The expected scale factor for n is computed based on the formula
    xTx = np.sum(np.square(x))
    nTn = np.sum(np.square(n))
    expected_scale_factor = ((xTx / nTn)**0.5) / (10**(SNR / 2))
    expected_result = expected_scale_factor * n
    
    assert np.allclose(result, expected_result), "Unexpected scaled noise for negative SNR"

def test_scale_noise_dimension_mismatch():
    n = np.ones(52)  # Incorrect dimension
    x = np.ones(53)
    
    with pytest.raises(AssertionError):
        scale_noise(n, x, 0)

    n = np.ones(53)
    x = np.ones(54)  # Incorrect dimension
    
    with pytest.raises(AssertionError):
        scale_noise(n, x, 0)



def test_create_colored_noise_mean(): #mean of the colored noise should not be zero
    noise = create_colored_noise(np.random.rand(53, 53), np.random.rand(53, 53), 1200)
    mean_noise = np.mean(noise, axis=1)
    assert not np.allclose(mean_noise, np.zeros(53))


def test_create_colored_noise_covariance(): #covariance of the noise should not be the identity
    noise = create_colored_noise(np.random.rand(53, 53), np.random.rand(53, 53), 1200)
    cov_noise = noise @ noise.T
    assert not np.allclose(cov_noise, np.eye(53))



# ?? other things to check for colored noise? 

"""def test_create_var_noises_len():
    A = np.random.rand(53, 1200)
    subjects = np.arange(1,11).astype(str)
    u_rate = 1
    nstd = 1.0
    burn = 100
    threshold = 0.0001
    noises = create_var_noise(A, subjects, threshold, u_rate, burn, 1200, nstd)
    assert len(noises) == len(subjects)"""



"""def test_create_var_first_noise_len():
    A = loadmat('/data/users2/jwardell1/undersampling-project/assets/data/VAR_data.mat')['A']    
    subjects = np.arange(1,11).astype(str)
    u_rate = 1
    nstd = 1.0
    burn = 100
    threshold = 0.0001
    noises = create_var_noise(A, subjects, threshold, u_rate, burn, 1200, nstd)
    sub = subjects[np.random.choice(10)]
    assert noises[sub].shape == (53, 1200)
"""


def preprocess_timecourse(tc_data):
    assert tc_data.shape[0] == 53, 'timecourse dimension 0 should be 53'
    data = detrend(tc_data, axis=1)   
    data = zscore(data, axis=1)
    return data

def test_preprocess_timecourse():
    # Create sample data
    np.random.seed(0)
    tc_data = np.random.rand(53, 100)  # 53 timecourses, each with 100 samples

    # Run preprocessing
    processed_data = preprocess_timecourse(tc_data)

    # Test 1: Check the dimensions of the processed data
    assert processed_data.shape == tc_data.shape, 'Processed data shape should match the input shape'

    # Test 2: Check that detrending removed linear trends
    for i in range(processed_data.shape[0]):
        # Check the linear trend removal: should be close to zero for detrended data
        assert np.allclose(np.polyfit(np.arange(processed_data.shape[1]), processed_data[i], 1)[0], 0, atol=1e-5), \
            f'Row {i} should have no linear trend after detrending'

    # Test 3: Check that z-scoring resulted in zero mean and unit variance
    for i in range(processed_data.shape[0]):
        assert np.allclose(np.mean(processed_data[i]), 0, atol=1e-5), f'Row {i} should have zero mean after z-scoring'
        assert np.allclose(np.std(processed_data[i]), 1, atol=1e-5), f'Row {i} should have unit variance after z-scoring'

    with pytest.raises(AssertionError):
        preprocess_timecourse(np.random.rand(52, 100))  # Incorrect shape





def test_parse_X_y_groups():
    data = []
    subjects = np.arange(1, 11).astype(str)  # 10 subjects
    window_types = ['SR1_Window', 'SR2_Window', 'Add_Window', 'Concat_Window']

    for subject in subjects:
        for target in [0, 1]:
            row = {'subject': subject, 'target': target}
            
            for window_type in window_types:
                if window_type == 'Add_Window':
                    row[window_type] = row['SR1_Window'] + row['SR2_Window']
                
                elif window_type == 'Concat':
                    row[window_type] = np.concat(row['SR1_Window'], row['SR2_Window'])

                else: 
                    row[window_type] = np.random.rand(1431)
            
            data.append(row)
    
    data_df = pd.DataFrame(data)

    sr = {
        'SR1_Window'    : 'SR1', 
        'SR2_Window'    : 'SR2', 
        'Add_Window'    : 'Add',
        'Concat_Window' : 'Concat',
    }

    for window_type in window_types:
        X, y, group = parse_X_y_groups(data_df, sr[window_type])

        sub_to_group = dict(zip(data_df['subject'], group))
        
        # Assertions
        assert len(X) == 20, f'Total number of data points for {window_type} should be num subs times 2'
        assert len(y) == 20, f'Total number of labels for {window_type} should be num subs times 2'
        assert set(y) == {'0', '1'}, f'Labels should be 0 or 1 for {window_type}'
        assert len(set(group)) == 10, f'Number of subjects should match number of groups for {window_type}'
        
        # Sum checks
        assert np.sum(np.array(data_df[window_type].tolist())) == np.sum(np.array(X)), f'Sum before and after extraction should be the same for {window_type}'

        
        # Order checks
        for i in range(20):
            assert np.allclose(X[i], data_df.iloc[i][window_type]), f'Data order should be preserved after extraction for {window_type}, idx {i}'
            subject_label = subjects[i//2]
            label = 0 if i % 2 == 0 else 1
            group_label = sub_to_group[subject_label]
            curr_data_obj = data_df[(data_df['subject'] == subject_label) & (data_df['target'] == label)][window_type]
            assert np.allclose(X[i], curr_data_obj.iloc[0]), 'data in X order should be preserved according to subject id'
            assert group[i] == group_label, f'group id {group[i]} should map to subject label {subject}'



            if sr[window_type] == 'Add':
                sr1 = curr_data_obj = data_df[(data_df['subject'] == subject_label) & (data_df['target'] == label)]['SR1_Window'].iloc[0]
                sr2 = curr_data_obj = data_df[(data_df['subject'] == subject_label) & (data_df['target'] == label)]['SR2_Window'].iloc[0]
                assert np.allclose(X[i], sr1 + sr2)
            elif sr[window_type] == 'Concat':
                sr1 = curr_data_obj = data_df[(data_df['subject'] == subject_label) & (data_df['target'] == label)]['SR1_Window'].iloc[0]
                sr2 = curr_data_obj = data_df[(data_df['subject'] == subject_label) & (data_df['target'] == label)]['SR2_Window'].iloc[0]
                assert np.allclose(X[i], np.concat(sr1, sr2))



def test_perform_windowing():
    pass


def test_load_timecourses():
    pass