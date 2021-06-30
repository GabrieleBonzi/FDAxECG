#%% GRAFICI RTSI CONFERENCE

t= np.arange(484)/SAMPLING_RATE

fig1 = plt.figure()
plt.title('Mean',fontsize=16)
plt.plot(t,t_mean_F_7_12, "r", label="STTC Mean")
plt.plot(t,t_mean_M_7_12, "b", label="NORM Mean")
plt.xlim([0,484/SAMPLING_RATE])
plt.legend()

plt.xlabel("Normalized time",fontsize=12)
plt.ylabel("Normalized amplitude",fontsize=12)

for v in np.mean(land_F_7_12[:,:5], axis=0):
    plt.axvline(x=v, color="k", lw=0.8)
    
for v in np.mean(land_M_7_12[:,:5], axis=0):
    plt.axvline(x=v, color="k", lw=0.8)


ax2=plt.twiny()

tickM=np.mean(land_M_7_12[:,:5], axis=0)
tickF=np.mean(land_F_7_12[:,:5], axis=0)
tick=(tickM+tickF)/2

ax2.set_xticks(tick)
ax2.set_xticklabels(["P", "Q", "R", "S", "T"])
ax2.set_xlim([0,484/SAMPLING_RATE])

#%%
fig, (ax1, ax2) = plt.subplots(1, 2)

fpca = FPCA(n_components=n)
fpca.fit(fd_M_7_12)

fpca.components_.plot(axes=ax1)
ax1.set_title("NORM Subjects")
ax1.set_xlabel("Normalized time",fontsize=12)
ax1.set_ylabel("[a.u.]",fontsize=12)

fpca = FPCA(n_components=n)
fpca.fit(fd_F_7_12)

fpca.components_.plot(axes=ax2)
plt.title("STTC Subjects")
ax2.set_xlabel("Normalized time",fontsize=12)
plt.legend(["PC1", "PC2", "PC3", "PC4"])

#%%

fpca = FPCA(n_components=n)
fpca.fit(fd_M_7_12)

pc, axs = plt.subplots(4)
for i in range(4):
    fpca.components_[i].plot(axes=axs[i], color='b')
    axs[i].set_title("PC" + str(i + 1), fontsize=16)
    axs[i].set_xlim([0,484/SAMPLING_RATE])
    axs[i].set_ylim([-4,6])
    axs[i].set_ylabel("[a.u.]",fontsize=12)
    
fpca = FPCA(n_components=n)
fpca.fit(fd_F_7_12)

for i in range(4):
    fpca.components_[i].plot(axes=axs[i], color='r')
    axs[i].set_title("PC" + str(i + 1))
    #axs[i].legend(["NORM", "STTC"])
    
axs[3].set_xlabel("Normalized time",fontsize=12)

#%%

S=D[-29:]
C=D[:-29]

plt.figure()
bplot=plt.boxplot((S,C),labels=["STTC","NORM"],patch_artist=True,widths=0.6)

colors = ['r','b']

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.title("Modified Band Depth (MBD)",fontsize=16)
plt.ylabel("Depth",fontsize=12)
plt.xlabel("Groups",fontsize=12)