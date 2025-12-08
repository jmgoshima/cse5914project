import React, { useCallback, useEffect, useMemo, useState } from "react";

const SAMPLE_RESULT = {
  header: "Nearest neighbors to input vector:",
  raw_output: `Nearest neighbors to input vector:
  1. New-Orleans,LA (score=0.9619)
  2. Dallas,TX (score=0.9610)
  3. Boston,MA (score=0.9574)
  4. Baltimore,MD (score=0.9535)
  5. Minneapolis-St.-Paul,MN-WI (score=0.9519)`,
  profile_payload: {
    Climate: 5.0,
    HousingCost: 0.0,
    HlthCare: 7.0,
    Crime: 8.5,
    Transp: 7.5,
    Educ: 7.0,
    Arts: 7.5,
    Recreat: 7.0,
    Econ: 7.5,
    Pop: 0.0,
  },
  reasoning: [
    {
      city: "New Orleans,LA",
      reason:
        "New Orleans matches well with your high preference for arts, recreation, and cultural life. It has a vibrant music and food scene, while offering moderate transportation and healthcare quality that align with your profile. Its warmer climate also fits your mid-range climate preference.",
    },
    {
      city: "Dallas, TX",
      reason:
        "Dallas provides strong economic opportunities and a well-developed transportation network, aligning with your high Econ and Transp scores. Housing costs are relatively affordable compared to other major cities, and the region offers a warmer climate consistent with your moderate climate tolerance.",
    },
    {
      city: "Boston, MA",
      reason:
        "Boston appeals to your interest in education and healthcare, both of which score highly in your profile. The city has extensive cultural and recreational options, and its strong economy matches your Econ preference, although housing costs tend to be higher.",
    },
    {
      city: "Baltimore, MD",
      reason:
        "Baltimore fits your high Crime tolerance but rewards you with solid healthcare, education, and access to cultural and recreational amenities. It offers urban energy with lower living costs than nearby major metros like Washington, D.C.",
    },
    {
      city: "Minneapolis-St.-Paul, MN-WI",
      reason:
        "The Twin Cities area aligns with your high arts, recreation, and education scores. It provides excellent healthcare and a strong economy. While the colder climate contrasts your mid-range climate score, the city compensates with high livability and civic infrastructure.",
    },
  ],
};

const UNSPLASH_BASE_URL = "https://api.unsplash.com/search/photos";
const UNSPLASH_ACCESS_KEY = process.env.REACT_APP_UNSPLASH_ACCESS_KEY || "";

const normaliseCityKey = (value = "") =>
  value
    .toString()
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");

const parseRawOutput = (rawOutput) => {
  if (!rawOutput || typeof rawOutput !== "string") {
    return [];
  }

  return rawOutput
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => /^\d+\./.test(line))
    .map((line, index) => {
      const match = line.match(/^\d+\.\s+(.*?)(?:\s+\(score=([0-9.]+)\))?$/i);
      if (!match) {
        return { label: line, score: null, id: `city-${index}` };
      }

      const [, nameWithState, scoreString] = match;
      const [name, state] = nameWithState
        .split(",")
        .map((part) => part?.trim());
      const score =
        typeof scoreString === "string" ? Number(scoreString) : null;

      return {
        id: `city-${index}`,
        name: name || nameWithState,
        state: state || "",
        label: nameWithState,
        score: Number.isFinite(score) ? score : null,
      };
    });
};

const CityRecommendations = ({
  isWaiting,
  result,
  enableDemoToggle = false,
}) => {
  const [demoState, setDemoState] = useState("waiting");
  const usingDemo = enableDemoToggle;
  const activeIsWaiting = usingDemo ? demoState === "waiting" : isWaiting;
  const activeResult =
    usingDemo && demoState === "ready" ? SAMPLE_RESULT : result;
  const [activeCityIndex, setActiveCityIndex] = useState(0);
  const [imageMap, setImageMap] = useState({});
  const [cityReasons, setCityReasons] = useState({});

  const demoControls = usingDemo ? (
    <div className="recommendations-demo-controls">
      <button type="button" onClick={() => setDemoState("waiting")}>
        Show waiting state
      </button>
      <button type="button" onClick={() => setDemoState("ready")}>
        Show result state
      </button>
    </div>
  ) : null;

  const header =
    activeResult && activeResult.header
      ? activeResult.header
      : "City recommendations";
  const cities = useMemo(() => {
    if (!activeResult) {
      return [];
    }

    if (Array.isArray(activeResult.cities) && activeResult.cities.length > 0) {
      return activeResult.cities;
    }

    return parseRawOutput(activeResult.raw_output);
  }, [activeResult]);
  const profileEntries = useMemo(() => {
    if (!activeResult || !activeResult.profile_payload) {
      return [];
    }

    return Object.entries(activeResult.profile_payload);
  }, [activeResult]);
  const reasoningMap = useMemo(() => {
    if (!activeResult || !Array.isArray(activeResult.reasoning)) {
      return {};
    }

    const map = {};
    activeResult.reasoning.forEach((entry) => {
      if (!entry?.city) {
        return;
      }

      const key = normaliseCityKey(entry.city);
      if (key) {
        map[key] = entry.reason;
      }
    });
    return map;
  }, [activeResult]);
  const cityIdentifiers = useMemo(
    () =>
      cities.map((city, index) => {
        const identifier =
          city.id || city.label || city.name || city.city || `city-${index}`;
        return identifier;
      }),
    [cities]
  );

  useEffect(() => {
    if (!activeResult || cities.length === 0) {
      setCityReasons({});
      return;
    }

    const mapping = {};

    cities.forEach((city, index) => {
      const identifier = cityIdentifiers[index];
      if (!identifier) {
        return;
      }

      const cityName =
        city.name || city.label || city.city || city.city_name || "";
      const stateName =
        city.state || city.state_name || city.region || city.state_code || "";

      const combinedKey = normaliseCityKey(`${cityName}${stateName}`);
      const fallbackKey = normaliseCityKey(cityName);

      const reasonFromMap =
        reasoningMap[combinedKey] || reasoningMap[fallbackKey] || null;
      const reasonFromCity =
        city.reason || city.summary || city.description || city.notes || null;

      mapping[identifier] = reasonFromMap || reasonFromCity || null;
    });

    setCityReasons(mapping);
  }, [activeResult, cities, cityIdentifiers, reasoningMap]);

  useEffect(() => {
    if (!activeResult) {
      setActiveCityIndex(0);
      return;
    }

    setActiveCityIndex(0);
  }, [activeResult]);

  useEffect(() => {
    if (cities.length === 0) {
      setActiveCityIndex(0);
      return;
    }

    setActiveCityIndex((prev) => {
      if (prev >= cities.length) {
        return cities.length - 1;
      }
      return prev;
    });
  }, [cities]);

  useEffect(() => {
    if (activeIsWaiting || cities.length === 0) {
      return;
    }

    const missingIdentifiers = cityIdentifiers.filter(
      (identifier) => identifier && imageMap[identifier] === undefined
    );
    if (missingIdentifiers.length === 0) {
      return;
    }

    let isCancelled = false;

    const fetchCityImages = async () => {
      const updates = {};

      await Promise.all(
        cities.map(async (city, index) => {
          const identifier = cityIdentifiers[index];
          if (!identifier || imageMap[identifier] !== undefined) {
            return;
          }

          const cityName =
            city.name || city.label || city.city || city.city_name || "";
          const stateName =
            city.state ||
            city.state_name ||
            city.region ||
            city.state_code ||
            "";
          const queryValue = [cityName, stateName]
            .filter(Boolean)
            .join(", ")
            .trim();
          const finalQuery =
            queryValue || cityName || stateName || "city skyline";

          if (!finalQuery) {
            updates[identifier] = null;
            return;
          }

          if (!UNSPLASH_ACCESS_KEY) {
            console.warn(`Unsplash access key not configured for city: ${finalQuery}`);
            updates[identifier] = null;
            return;
          }

          const url = new URL(UNSPLASH_BASE_URL);
          url.searchParams.set("query", finalQuery);
          url.searchParams.set("per_page", "1");
          url.searchParams.set("orientation", "landscape");

          try {
            const response = await fetch(url.toString(), {
              headers: {
                Authorization: `Client-ID ${UNSPLASH_ACCESS_KEY}`,
                "Accept-Version": "v1",
              },
            });

            if (!response.ok) {
              console.error(`Unsplash API error for "${finalQuery}": ${response.status} ${response.statusText}`);
              updates[identifier] = null;
              return;
            }

            const payload = await response.json();
            const imageUrl = payload?.results?.[0]?.urls?.regular || null;
            
            if (!imageUrl) {
              console.warn(`No image found in Unsplash response for: ${finalQuery}`, payload);
            }

            updates[identifier] = imageUrl;
          } catch (err) {
            console.error(`Failed to fetch Unsplash image for "${finalQuery}":`, err);
            updates[identifier] = null;
          }
        })
      );

      if (!isCancelled && Object.keys(updates).length > 0) {
        setImageMap((prev) => ({ ...prev, ...updates }));
      }
    };

    fetchCityImages();

    return () => {
      isCancelled = true;
    };
  }, [activeIsWaiting, cities, cityIdentifiers, imageMap]);

  const cityCount = cities.length;
  const showNavigation = cityCount > 1;

  const goToPreviousCity = useCallback(() => {
    if (cityCount === 0) {
      return;
    }

    setActiveCityIndex((prev) => (prev === 0 ? cityCount - 1 : prev - 1));
  }, [cityCount]);

  const goToNextCity = useCallback(() => {
    if (cityCount === 0) {
      return;
    }

    setActiveCityIndex((prev) => (prev === cityCount - 1 ? 0 : prev + 1));
  }, [cityCount]);

  const activeCity =
    cityCount > 0 ? cities[activeCityIndex] || cities[0] : null;
  const activeIdentifier =
    cityCount > 0
      ? cityIdentifiers[activeCityIndex] || cityIdentifiers[0]
      : null;
  const activeImageUrl =
    activeIdentifier && imageMap[activeIdentifier] !== undefined
      ? imageMap[activeIdentifier]
      : null;
  const cityName =
    activeCity?.name || activeCity?.label || activeCity?.city || "Unnamed city";
  const stateSuffix =
    activeCity?.state || activeCity?.state_code || activeCity?.region || "";
  const displayName = stateSuffix ? `${cityName}, ${stateSuffix}` : cityName;
  const score =
    typeof activeCity?.score === "number"
      ? activeCity.score
      : typeof activeCity?.value === "number"
      ? activeCity.value
      : null;
  const activeReason =
    (activeIdentifier && cityReasons[activeIdentifier] !== undefined
      ? cityReasons[activeIdentifier]
      : null) ??
    activeCity?.reason ??
    activeCity?.summary ??
    activeCity?.description ??
    activeCity?.notes ??
    null;

  if (activeIsWaiting) {
    return (
      <div className="recommendations-card">
        {demoControls}
        <h2>City recommendations</h2>
        <p className="recommendations-hint">Waiting for more info…</p>
        <div className="city-placeholder">
          <span className="city-placeholder-dot" />
          <span className="city-placeholder-text">
            Hang tight—your tailored matches will pop in as soon as the
            assistant finishes.
          </span>
        </div>
      </div>
    );
  }

  if (!activeResult) {
    return (
      <div className="recommendations-card">
        {demoControls}
        <h2>City recommendations</h2>
        <p className="recommendations-hint">
          Start chatting to generate city recommendations tailored to your
          profile.
        </p>
        <div className="city-placeholder">
          <span className="city-placeholder-dot" />
          <span className="city-placeholder-text">
            No recommendations yet—ask the assistant about your ideal city
            profile.
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="recommendations-card">
      <h2>{header}</h2>

      {cityCount > 0 ? (
        <div className="city-carousel">
          {showNavigation && (
            <button
              type="button"
              className="city-carousel-button city-carousel-button--prev"
              onClick={goToPreviousCity}
              aria-label="View previous city recommendation"
            >
              <span aria-hidden="true">‹</span>
            </button>
          )}

          <div className="city-carousel-track">
            <div className="city-item" role="group" aria-label={displayName}>
              <div className="city-item-media">
                {activeImageUrl ? (
                  <img
                    src={activeImageUrl}
                    alt={`${displayName} skyline`}
                    loading="lazy"
                  />
                ) : (
                  <div className="city-image--placeholder">No image</div>
                )}
              </div>

              <div className="city-item-content">
                <div className="city-item-header">
                  <div className="city-name">{displayName}</div>
                  {score !== null && (
                    <div className="city-score">
                      <span className="city-score-value">
                        {score.toFixed(3)}
                      </span>
                      <span className="city-score-label">match score</span>
                    </div>
                  )}
                </div>

                {activeReason && <p className="city-reason">{activeReason}</p>}
              </div>
            </div>

            {showNavigation && (
              <div
                className="city-carousel-indicator"
                role="status"
                aria-live="polite"
              >
                <span className="city-carousel-index">
                  {String(activeCityIndex + 1).padStart(2, "0")}
                </span>
                <span className="city-carousel-separator" />
                <span className="city-carousel-total">
                  {String(cityCount).padStart(2, "0")}
                </span>
              </div>
            )}
          </div>

          {showNavigation && (
            <button
              type="button"
              className="city-carousel-button city-carousel-button--next"
              onClick={goToNextCity}
              aria-label="View next city recommendation"
            >
              <span aria-hidden="true">›</span>
            </button>
          )}
        </div>
      ) : (
        <div className="city-placeholder">
          <span className="city-placeholder-dot" />
          <span className="city-placeholder-text">
            The current response didn’t include any ranked cities.
          </span>
        </div>
      )}

      {profileEntries.length > 0 && (
        <div className="city-profile">
          <h3>Profile Highlights</h3>
          <dl className="city-profile-grid">
            {profileEntries.map(([key, value]) => (
              <React.Fragment key={key}>
                <dt>
                  <span className="city-profile-label">{key}</span>
                  <span className="city-profile-bar">
                    <span
                      className="city-profile-bar-fill"
                      style={{
                        width: `${Math.min(Number(value) * 10, 100)}%`,
                      }}
                    />
                  </span>

                  <span className="city-profile-value-under">
                    {typeof value === "number" ? value.toFixed(1) : value}
                  </span>
                </dt>
              </React.Fragment>
            ))}
          </dl>
        </div>
      )}
    </div>
  );
};

export default CityRecommendations;
