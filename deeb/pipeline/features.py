# For computing AR features
df=pd.DataFrame()
for subject in subject_dict:
    #print("SUbject", subject)
    epochs=subject_dict[subject]
    epochs_data=epochs.get_data()
    #print(epochs.get_data().shape)
    order=6
    #print("epochs data length", len(epochs_data))
    #for ord in order:
    for i in range(len(epochs_data)):
        dictemp={}
        for j in range(len(epochs_data[i])):
            rho, sigma = sm.regression.yule_walker(epochs_data[i][j], order=order, method="mle")
            #print("rho value", rho)
            #ar_order_opt = np.where(np.abs(rho) > 1e-6)
            #print(ar_order_opt)
            dictemp['Subject']=subject
            #print(epochs[i].event_id.values())
            dictemp['Event_id']=list(epochs[i].event_id.values())[0]
            #print(dictemp)
        
            first=epochs.ch_names[j]
            for d in range(order):
                colum_name=first+"-AR"+str(d+1)
                dictemp[colum_name]=rho[d]  
        data_step = [dictemp]
        df=df.append(data_step,ignore_index=True)
        #print(df)

print("========================================================================================================================")
#df_psd.max().max()
df_psd=pd.DataFrame()
for subject in subject_dict:
    #if(loc==len(df)):
     #   break
    #else:
    #print("SUbject", subject)
    #subject=str(subject)
    epochs=subject_dict[subject]
        #epochs_data=epochs.get_data()
    #print(epochs.get_data().shape)
    tmax=epochs.tmax
    tmin=epochs.tmin
    #fmin=epochs.fmin
    #fmax=epochs.fmax
    #print(fmin)
    #print(fmax)
    
    sfreq=epochs.info['sfreq']
        # specific frequency bands
    FREQ_BANDS = {"delta" : [1,4],
                    "theta" : [4,8],
                    "alpha" : [8, 12],
                    "beta" : [12,30],
                    "gamma" : [30, 50]}
    
    spectrum=epochs.compute_psd(method="welch", n_fft=int(sfreq * (tmax - tmin)),
                           n_overlap=0, n_per_seg=None, fmin=1, fmax=50, tmin=tmin, tmax=tmax, verbose=False)
    
    psds, freqs=spectrum.get_data(return_freqs=True)
    #spectrum.co
    # Normalize the PSDs
    #psds /= np.sum(psds, axis=-1, keepdims=True)
    #print(psds.shape)
    #X = []
    #for fmin, fmax in FREQ_BANDS.values():
     #   psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
      #  print(fmin," ", fmax, " ", psds_band[2])
        #print("bands before reshpe", psds_band[0])
        #print("bands after reshpe", psds_band.reshape(len(psds), -1)[0])
       # X.append(psds_band.reshape(len(psds), -1))
        
    
    #print(np.concatenate(X, axis=1)[0])
    #psd_freq_bands=np.concatenate(X, axis=1).reshape(len(psds), psds.shape[1], 5)
    
    #features[subject]=psd_freq_bands
    #features
    #print(psd_freq_bands.shape)
    
    for i in range(len(psds)):
        features={}
        #print(i)
        for j in range(len(psds[i])):
            #print(psds[i][j])
            welch_psd=psds[i][j]
            X=[]
            for fmin, fmax in FREQ_BANDS.values():
                psds_band=welch_psd[(freqs >= fmin) & (freqs < fmax)].mean()
                #print(psds_band)
                X.append(psds_band)
                
            features['Subject']=subject
            features['Event_id_PSD']=list(epochs[i].event_id.values())[0]
            
            channel=epochs.ch_names[j]
            for d in range(len(X)):
                band_name=[*FREQ_BANDS][d]
                colum_name=channel+"-"+band_name
                features[colum_name]=X[d]
                
            #loc=loc+1
            #print(df)
        data_step = [features]
        df_psd=df_psd.append(data_step,ignore_index=True)
    #print(df)
#display(df_psd)



