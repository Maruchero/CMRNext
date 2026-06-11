- [x] Verificare il funzionamento di evaluate_flow_calibration.py
- [x] Creare script per convertire immagini e pointcloud UR in formato accettabile da evaluate_flow_calibration.py
- [ ] plottare ad ogni iterazione su weights per vedere se ci sono differenze
- [x] aggiungere undistortion dell'immagine (o distortion lidar points)
- [ ] plottare a sinistra bev e a destra immagine, poi tracciare delle linee rosse per le corrispondenze e valutare visivamente la correttezza delle stesse
- [ ] se le feature estratte sono sensate valutare la sostituzione del pnp con qualcosa di più robusto (vedi GTSAM)
 and have the pointcloud on the left, the image on the right and some red lines showing where each lidar feature falls in the image