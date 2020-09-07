"""A module, which contains utilities for generating standard plots for the report."""

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import roc_curve, auc
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_plots_for_rooms(data, df_meta, folder=None, verbose=False):
    room_ids = [np.unique(df_meta[df_meta.room == room_number].receiver_id) for room_number in np.unique(df_meta.room.values)]
    ids = np.concatenate(room_ids)

    img = np.zeros((len(ids),len(ids)))
    img_dist = np.zeros((len(ids),len(ids)))
    img_rss = np.zeros((len(ids),len(ids)))

    for i, row in df_meta.iterrows():
        id1 = np.where(ids == row.receiver_id)
        id2 = np.where(ids == row.sender_id)
        img[id1,id2] = row.room
        img_dist[id1,id2] += (1./data[i]['dist']).sum()
        img_rss[id1,id2] += (-data[i]['rss']).sum()

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img/5, cmap='jet')
    plt.xlabel("Users")
    plt.ylabel("Users")
    plt.title("Overview of users and rooms")

    plt.subplot(1,3,2)
    ax = plt.gca()
    im = ax.imshow(img_dist, cmap='Reds')
    plt.xlabel("Users")
    plt.ylabel("Users")
    plt.title("Sum of inverse Distances")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.subplot(1,3,3)
    ax = plt.gca()
    im = ax.imshow(img_rss, cmap='Reds')
    plt.xlabel("Users")
    plt.ylabel("Users")
    plt.title("Negative Sum of RSS values")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()

    if verbose:
        plt.show()
    else:
        plt.savefig(folder+'experimental_setup.pdf', formatt='pdf')
        plt.close()

    u_dist = np.triu(img_dist).flatten() # pylint: disable=no-member
    u_rss = np.triu(img_rss).flatten() # pylint: disable=no-member
    l_dist = np.tril(img_dist).flatten() # pylint: disable=no-member
    l_rss = np.tril(img_rss).flatten() # pylint: disable=no-member

    plt.scatter(u_dist, u_rss, label='receiver', alpha=.5,c='k',marker='o')
    plt.scatter(l_dist, l_rss, label='sender', alpha=.5,c='k',marker='x')
    for i in range(img_dist.shape[0]): # pylint: disable=unsubscriptable-object
        for j in range(img_dist.shape[1]): # pylint: disable=unsubscriptable-object
            plt.plot([img_dist[i,j], img_dist[j,i]], [img_rss[i,j], img_rss[j,i]], c='r', alpha=1)
    plt.xlabel('sum of proximities')
    plt.legend()
    plt.ylabel('sum of signal strenghts')
    plt.tight_layout()

    if verbose:
        plt.show()
    else:
        plt.savefig(folder+'receiver_sender_scatter.pdf', formatt='pdf')
        plt.close()

def visualize_pair(ax1,raw_user1, raw_user2, user1, user2):
    ax2 = ax1.twinx()
    # user 1
    user1_color='g'
    user1_name=raw_user1['receiver']['phone_model']
    # user1 rss
    ax1.plot(pd.to_datetime(raw_user1['time'], unit='ms'), raw_user1['rss'], alpha=.5, label=user1_name+' raw ', c=user1_color)
    ax1.plot(pd.to_datetime(user1['time'], unit='ms'), user1['rss'], alpha=1, label=user1_name+' resampled ', c=user1_color)
    # user1 distance
    ax2.plot(pd.to_datetime(raw_user1['time'], unit='ms'), raw_user1['dist'], c='r', label='distance')
    ax2.plot(pd.to_datetime(raw_user2['time'], unit='ms'), raw_user2['dist'], c='r', label='distance')
    # user 2
    user2_color='b'
    user2_name=raw_user2['receiver']['phone_model']
    # user1 rss
    ax1.plot(pd.to_datetime(raw_user2['time'], unit='ms'), raw_user2['rss'], alpha=.5, label=user2_name+' raw ', c=user2_color)
    ax1.plot(pd.to_datetime(user2['time'], unit='ms'), user2['rss'], alpha=1, label=user2_name+' resampled ', c=user2_color)

    ax1.set_ylabel('RSS as db', color='g')
    ax1.legend(loc='upper left', prop={'size': 10})
    ax1.set_ylim(-130, -10)
    ax2.set_ylabel('Distance as cm', color='r')
    ax2.set_ylim(40., 500.)
    ax2.legend(loc='upper right', prop={'size': 10})
    ax1.set_title('Scenario {}'.format(raw_user1['scenario']))

def random_samples(data, resampled_data, df_meta, folder=None, verbose=False, expected_numer_of_experiments=4):
    m=2
    fig, axarr = plt.subplots(m,m, figsize=(m*8,m*4))
    for i in range(m*m):
        # find appropriate pair
        while True:
            random_room = np.random.choice(np.unique(df_meta.room.values))
            selection = df_meta[(df_meta.room == random_room)] 
            random_exp = np.random.choice(np.unique(selection.exp.values))  
            selection = selection[selection.exp == random_exp] 

            user1_id = np.random.choice(selection.receiver_id.values)
            selection = selection[selection.receiver_id == user1_id]
            user2_id = np.random.choice(selection.sender_id.values)

            df1 = df_meta[(
            (df_meta.receiver_id == user1_id)&
            (df_meta.sender_id == user2_id)&
            (df_meta.room == random_room)&
            (df_meta.exp == random_exp)
                   )]

            df2 = df_meta[(
            (df_meta.receiver_id == user2_id)&
            (df_meta.sender_id == user1_id)&
            (df_meta.room == random_room)&
            (df_meta.exp == random_exp)
                   )]

            if ((len(df1) > 0) & (len(df2) > 0)):
                break

        visualize_pair(
            axes[i//m, i%m],
            data[int(df1.index[0])],
            data[int(df2.index[0])],
            resampled_data[int(df1.index[0])],
            resampled_data[int(df2.index[0])],
        )
    fig.tight_layout()

    if verbose:
        plt.show()
    else:
        plt.savefig(folder+'random_samples.pdf', formatt='pdf')
        plt.close()

def plot_roc(y_true, y_pred, title):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=title+' AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    return roc_auc, fpr, tpr, thresholds

def make_new_latex_command(name, value, dtype):
    prefix = '\\newcommand{\\'
    if dtype == 'str':
        return prefix + name + '}{' + value + '}\n'
    elif dtype == 'numeric':
        return prefix + name + '}{$' + str(value) + '$}\n'

def number_of_connections(df_meta):
    sum = 0
    for room in np.unique(df_meta.room):
        users = np.unique(df_meta[df_meta.room == room].receiver_id)
        sum += len(users)*(len(users)-1)
    return sum

def total_recording_length(data):
    return np.round(sum([len(d) for d in data])/(60.*60.),2)

def generate_statistics(data, df_meta, folder=None, verbose=False):
    stats_string = ''
    # day
    stats_string += make_new_latex_command('Day', folder, 'str')

    # number of participants
    participants = np.unique(df_meta.receiver_id)
    stats_string += make_new_latex_command('NumberOfPatients', len(participants), 'numeric')

    # number of phones per model
    #stats_string += make_new_latex_command('NumberOfPhonesPerModel', 0 , 'numeric')

    # number of connections
    stats_string += make_new_latex_command('NumberOfConnections', number_of_connections(df_meta) , 'numeric')

    # number of samples
    stats_string += make_new_latex_command('NumberOfSamples', len(data) , 'numeric')

    # total recording length
    stats_string += make_new_latex_command('TotalRecordingLength', total_recording_length(data) , 'numeric')

    for i, room_name in enumerate(['One', 'Two', 'Three', 'Four', 'Five']):
        # number of samples in room i
        stats_string += make_new_latex_command('NumberOfDevicesRoom'+room_name, len(np.unique(df_meta[df_meta.room ==i+1].receiver_id)) , 'numeric')
        # number of samples in room i
        stats_string += make_new_latex_command('NumberOfSamplesRoom'+room_name, len(df_meta[df_meta.room ==i+1]) , 'numeric')

    # device statistics
    device_counts = df_meta.groupby('receiver_id').min()['receiver_model'].value_counts()
    device_string = '\\begin{itemize} \n'
    for (device, count) in device_counts.iteritems():
        device_string += '\\item ' + str(count) + ' ' + str(device.replace('samsung_SM', 'Samsung SM') + '\n')
    device_string += '\\end{itemize} \n'
    stats_string += make_new_latex_command('DevicesList', device_string , 'str')

    if verbose:
        print(stats_string)
        print(folder+"stats.tex")
    else:
        output_file = open(folder+"stats.tex", "w")
        output_file.write(stats_string)
        output_file.close()
