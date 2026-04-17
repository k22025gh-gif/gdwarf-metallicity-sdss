-- sdss_gstars_query.sql
-- ----------------------
-- CasJobs query (SDSS DR17) used to generate the input star catalog.
-- Run this on https://skyserver.sdss.org/casjobs/ (context: DR17)
-- and download the result as a CSV file.
--
-- Selects G-type main-sequence stars with:
--   - SSPP stellar parameters available (sppParams join)
--   - Median r-band S/N >= 15
--   - PSF r-band magnitude 10–15 (bright, well-exposed stars)
--   - Ordered by brightness (brightest first)
--
-- Output columns used by preprocess_sdss.py:
--   plate, mjd, fiber          → spectrum identifier for astroquery fetch
--   ra, dec                    → sky coordinates
--   snMedian_r                 → SDSS pipeline S/N in r band
--   psfMag_r                   → PSF magnitude in r band
--   TEFFADOP, TEFFADOPUNC      → SSPP effective temperature and uncertainty
--   LOGGADOP, LOGGADOPUNC      → SSPP surface gravity and uncertainty
--   FEHADOP, FEHADOPUNC        → SSPP metallicity and uncertainty
--   ELODIERVFINAL              → SSPP radial velocity (km/s, heliocentric)
--   ELODIERVFINALERR           → uncertainty on ELODIERVFINAL (km/s)

SELECT TOP 2000
    s.plate,
    s.mjd,
    s.fiberID                AS fiber,
    s.ra,
    s.dec,
    s.snMedian_r,
    p.psfMag_r,
    sp.TEFFADOP,
    sp.TEFFADOPUNC,
    sp.LOGGADOP,
    sp.LOGGADOPUNC,
    sp.FEHADOP,
    sp.FEHADOPUNC,
    sp.ELODIERVFINAL,
    sp.ELODIERVFINALERR
FROM SpecObjAll AS s
JOIN PhotoObj  AS p
    ON s.bestObjID = p.objID
JOIN sppParams AS sp
    ON s.specObjID = sp.SPECOBJID
WHERE s.class      = 'STAR'
  AND s.subClass  LIKE 'G%'
  AND s.snMedian_r >= 15
  AND p.psfMag_r  BETWEEN 10 AND 15
ORDER BY p.psfMag_r ASC
