import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
from pathlib import Path

def make_synthetic_examples(base: Path):
    base.mkdir(parents=True, exist_ok=True)
    crs = "EPSG:26917"
    domain = gpd.GeoDataFrame({"id":[1]}, geometry=[Polygon([(0,0),(2000,0),(2000,2000),(0,2000)])], crs=crs)
    domain.to_file(base / "domain.gpkg", driver="GPKG")

    polys = []
    pids = []
    w = 1000.0; h = 1000.0/1.5
    pid = 1
    for i in range(2):
        for j in range(3):
            x0 = i*w
            y0 = j*h
            polys.append(Polygon([(x0,y0),(x0+w,y0),(x0+w,y0+h),(x0,y0+h)]))
            pids.append(pid)
            pid += 1
    parcels = gpd.GeoDataFrame({"pid":pids}, geometry=polys, crs=crs)
    parcels.to_file(base / "parcels.gpkg", driver="GPKG")

    outlets = gpd.GeoDataFrame({"oid":[101,102]},
                               geometry=[Point(1800,100), Point(1800,1800)],
                               crs=crs)
    outlets.to_file(base / "outlet_loc.gpkg", driver="GPKG")

    po = pd.DataFrame({"pid":pids, "oids":["101,102"]*len(pids)})
    po.to_csv(base / "parcel_out.csv", index=False)

    ups = []
    for i in range(1,7):
        if i in (1,2):
            ups.append("")
        elif i in (3,4):
            ups.append("1,2")
        elif i in (5,6):
            ups.append("3,4")
    pu = pd.DataFrame({"pid":pids, "pid_up":ups})
    pu.to_csv(base / "parcel_up.csv", index=False)

    pp = pd.DataFrame({"pid":pids, "probability":[1.0/len(pids)]*len(pids)})
    pp.to_csv(base / "parcel_p.csv", index=False)

    poll = []
    for pid in pids:
        for pol in ["nitrogen","phosphorus","sediment"]:
            if pol=="nitrogen":
                poll.append({"pid":pid,"pollutant":pol,"min":5,"max":30,"mean":15,"p50":15})
            elif pol=="phosphorus":
                poll.append({"pid":pid,"pollutant":pol,"min":0.5,"max":5.0,"mean":2.0,"p50":2.0})
            else:
                poll.append({"pid":pid,"pollutant":pol,"min":50,"max":400,"mean":150,"p50":150})
    pd.DataFrame(poll).to_csv(base / "pollutant_yield.csv", index=False)

    eff = []
    for cps in [340,329,590,412,656]:
        for pol in ["nitrogen","phosphorus","sediment"]:
            if cps==656 and pol=="sediment":
                eff.append({"cps":cps,"pollutant":pol,"min":0.3,"max":0.8,"mean":0.6,"p50":0.6})
            elif cps==412:
                eff.append({"cps":cps,"pollutant":pol,"min":0.05,"max":0.3,"mean":0.15,"p50":0.15})
            else:
                eff.append({"cps":cps,"pollutant":pol,"min":0.05,"max":0.5,"mean":0.2,"p50":0.2})
    pd.DataFrame(eff).to_csv(base / "bmp_efficiency.csv", index=False)

    cost = [
        {"cps":340,"unit":"USD/ha","min":25,"max":75,"mean":50,"p50":50},
        {"cps":329,"unit":"USD/ha","min":10,"max":60,"mean":35,"p50":35},
        {"cps":590,"unit":"USD/ha","min":30,"max":120,"mean":60,"p50":60},
        {"cps":412,"unit":"USD/ha","min":1500,"max":4000,"mean":2500,"p50":2500},
        {"cps":656,"unit":"USD/ha","min":5000,"max":30000,"mean":15000,"p50":15000},
    ]
    pd.DataFrame(cost).to_csv(base / "bmp_cost.csv", index=False)

    ot = []
    om = []
    for oid in [101,102]:
        for pol in ["nitrogen","phosphorus","sediment"]:
            ot.append({"oid":oid,"pollutant":pol,"target":1000})
            om.append({"oid":oid,"pollutant":pol,"mean":5000})
    pd.DataFrame(ot).to_csv(base / "outlet_target.csv", index=False)
    pd.DataFrame(om).to_csv(base / "outlet_mean.csv", index=False)

    dr = []
    for pid in pids:
        for oid in [101,102]:
            dr.append({"pid":pid,"oid":oid,"sdr_f_to_s":0.9,"sdr_s_to_o":0.95,"ndr_f_to_s":0.8,"ndr_s_to_o":0.9})
    pd.DataFrame(dr).to_csv(base / "delivery_ratios.csv", index=False)
