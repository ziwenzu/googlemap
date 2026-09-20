# Government-place classification codebook, draft v0.1

## Unit

One Google place listing identified by gmap_id. A listing is not automatically a
physical facility: one facility can have multiple listings, and one building can
house several facilities.

## Core decision

A core government place must satisfy all three conditions:

1. the listing represents a government agency or a facility delivering a delegated
   public administrative service;
2. members of the public can have a qualifying encounter with that facility;
3. a review can plausibly describe that encounter rather than only a landmark,
   geographic area, museum, memorial, or unrelated colocated business.

## Fields

### government_level

- federal
- state
- county
- municipal
- special_district
- mixed
- uncertain

### ownership_status

- public
- contracted_public_service
- quasi_public
- private_regulated_only
- uncertain

Private entities are not treated as government merely because they are regulated.

### encounter_mode

- transaction: license, benefit, filing, mailing, permit, record, payment
- adjudication: court, hearing, appeal, enforcement decision
- emergency: police, fire, emergency response
- community: library, community program, public meeting
- amenity: park, recreation, visitor attraction
- nonpublic: administrative building with no public-facing service

### public_facing

- yes
- no
- uncertain

### analytic_tier

- core: clear administrative or transactional service
- secondary: politically important but distinct observation process
- exclude: no qualifying public-service encounter for this study
- uncertain: requires adjudication

## Common exclusions

- UPS, FedEx, private mailbox, shipping, and postal-annex stores;
- driving schools, private tag agents, and car dealers;
- tax preparers, accountants, and title companies;
- nonprofit or private providers absent a separately defined delegated-service
  research question;
- police/fire museums, memorials, and historic markers;
- collection boxes and unattended infrastructure;
- neighborhoods, cities, oceans, and other geographic features;
- restaurants, parking garages, and shops merely located in government buildings;
- campaign offices and political-party organizations;
- duplicate or spoofed listings until resolved.

## Authenticity is a separate label

This codebook classifies the place, not the truth of individual reviews. Do not
exclude a government listing because its reviews look political, coordinated, or
implausible. Those are review-level measurement questions.

