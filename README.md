# AR Earthquake Visualization

This interactive AR app visualizes real earthquake data from Indonesia’s Sumatra region on a 3D map overlay. Built using Unity and Mapbox for the Meta Quest headset, it features immersive interaction with 25 years of earthquake activity (2000–2025).

### Haptic Feedback
Each earthquake site in the AR map emits haptic vibrations based on real seismic data. When the user places their controller over an earthquake epicenter, the app:

- Plays an audio clip of the quake
- Triggers haptics reflecting the quake’s seismographic waveform and magnitude

The waveforms were extracted using the SAGE Wilber 3 tool and processed into `.haptic` files via a custom Python pipeline that matched waveform amplitude to vibration strength.

### Final APK
The latest build (`Eearthquake Visualization Final.apk`) is included in this repo and tracked via Git LFS.

---

**Data sources:**  
- USGS Earthquake Catalog  
- SAGE Wilber 3 (waveforms and audio)
