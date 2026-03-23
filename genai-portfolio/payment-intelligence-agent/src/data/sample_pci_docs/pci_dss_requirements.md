# PCI DSS v4.0 Requirements Overview

## Introduction

The Payment Card Industry Data Security Standard (PCI DSS) v4.0 was published in March 2022 and becomes mandatory on March 31, 2025. It provides a baseline of technical and operational requirements designed to protect cardholder data. PCI DSS applies to all entities that store, process, or transmit cardholder data, and to all entities that could affect the security of the cardholder data environment (CDE).

PCI DSS v4.0 introduces a customized approach as an alternative to the traditional defined approach, allowing organizations more flexibility in how they meet security objectives, provided they can demonstrate effectiveness through rigorous risk analysis.

## Requirement 1: Install and Maintain Network Security Controls

Network security controls (NSCs), such as firewalls and other network security technologies, are critical points of policy enforcement that control network traffic between two or more logical or physical network segments. Traditionally, this function was performed by physical firewalls; however, cloud environments, virtual networks, and other technologies now provide filtering capabilities similar to traditional firewalls.

Key sub-requirements:
- **1.1** Processes and mechanisms for installing and maintaining network security controls are defined and understood by all affected parties.
- **1.2** Network security controls (NSCs) are configured and maintained to restrict inbound and outbound traffic to that which is necessary for the cardholder data environment.
- **1.3** Network access to and from the cardholder data environment is restricted. A DMZ must be implemented to limit inbound traffic to only system components that provide authorized publicly accessible services.
- **1.4** Network connections between trusted and untrusted networks are controlled, including controls to restrict traffic between wireless networks and the CDE.
- **1.5** Risks to the CDE from computing devices that connect to both untrusted networks and the CDE are mitigated.

## Requirement 2: Apply Secure Configurations to All System Components

Malicious individuals often use vendor default passwords and other vendor default settings to compromise systems. These passwords and settings are well known in hacker communities and are easily determined by automated scanning tools.

Key sub-requirements:
- **2.1** Processes and mechanisms for applying secure configurations to all system components are defined and understood.
- **2.2** System components are configured and managed securely. Vendor-supplied defaults for system passwords and other security parameters must be changed before installation on the network. Unnecessary default accounts must be removed or disabled.
- **2.3** Wireless environments are configured and managed securely, including changing wireless vendor defaults for encryption keys, passwords, and SNMP community strings.

## Requirement 3: Protect Stored Account Data

Protection methods such as encryption, truncation, masking, and hashing are critical components of cardholder data protection. If an intruder circumvents other security controls and gains access to encrypted data, without the proper cryptographic keys, the data is unreadable and unusable to that person.

Key sub-requirements:
- **3.1** Processes and mechanisms for protecting stored account data are defined and understood.
- **3.2** Storage of account data is kept to a minimum through implementation of data retention and disposal policies.
- **3.3** Sensitive authentication data (SAD) is not stored after authorization, even if encrypted. This includes the full contents of any track, card verification codes, and PINs/PIN blocks.
- **3.4** Access to displays of full PAN and ability to copy cardholder data are restricted to those with a documented business need.
- **3.5** Primary account number (PAN) is secured wherever it is stored. PAN must be rendered unreadable using strong cryptography with associated key-management processes, one-way hashes, truncation, or index tokens.
- **3.6** Cryptographic keys used to protect stored account data are secured. Access to keys must be restricted to the fewest number of custodians necessary.
- **3.7** Key management processes and procedures covering the entire key lifecycle are implemented, including generation, distribution, storage, rotation, revocation, and destruction.

## Requirement 4: Protect Cardholder Data with Strong Cryptography During Transmission

Sensitive information must be encrypted during transmission over open, public networks because it is easy for a malicious individual to intercept and/or divert data while in transit.

Key sub-requirements:
- **4.1** Processes and mechanisms for protecting cardholder data with strong cryptography during transmission over open, public networks are defined and documented.
- **4.2** PAN is protected with strong cryptography during transmission. Only trusted keys and certificates are accepted. TLS 1.2 or higher must be used; SSL and early TLS are not considered strong cryptography. Certificates must be confirmed as valid and not expired or revoked.

## Requirement 5: Protect All Systems and Networks from Malicious Software

Malicious software, commonly referred to as "malware," including viruses, worms, and Trojans, enters the network during many business-approved activities including employee email and use of the Internet, mobile computers, and storage devices, resulting in the exploitation of system vulnerabilities.

Key sub-requirements:
- **5.1** Processes and mechanisms for protecting all systems and networks from malicious software are defined and understood.
- **5.2** Malicious software (malware) is prevented, or detected and addressed. Anti-malware solutions must be deployed on all system components commonly affected by malware.
- **5.3** Anti-malware mechanisms and processes are active, maintained, and monitored. Solutions must be kept current via automatic updates and periodic scans.
- **5.4** Anti-phishing mechanisms protect users against phishing attacks, including both email and web-based phishing.

## Requirement 6: Develop and Maintain Secure Systems and Software

Unpatched vulnerabilities are exploited by malicious individuals to gain privileged access to systems. Many of these vulnerabilities are fixed by vendor-provided security patches, which must be installed within a defined time frame.

Key sub-requirements:
- **6.1** Processes and mechanisms for developing and maintaining secure systems and software are defined and understood.
- **6.2** Bespoke and custom software is developed securely, applying secure coding guidelines and practices throughout the software development lifecycle.
- **6.3** Security vulnerabilities are identified and addressed. Critical vulnerabilities must be patched within one month of release. A risk ranking process must be used to prioritize patch installation.
- **6.4** Public-facing web applications are protected against attacks through web application firewalls (WAFs) or similar technology.
- **6.5** Changes to all system components are managed securely through a formal change control process.

## Requirement 7: Restrict Access to System Components and Cardholder Data by Business Need to Know

To ensure critical data can only be accessed by authorized personnel, systems and processes must be in place to limit access based on need to know and according to job responsibilities. Need to know means access rights are granted to only the least amount of data and privileges needed to perform a job.

Key sub-requirements:
- **7.1** Processes and mechanisms for restricting access to system components and cardholder data by business need to know are defined and understood.
- **7.2** Access to system components and data is appropriately defined and assigned based on job classification and function, implementing a role-based access control (RBAC) system.
- **7.3** Access to system components and data is managed via an access control system(s) that restricts access based on a user's need to know and is set to "deny all" unless specifically allowed.

## Requirement 8: Identify Users and Authenticate Access to System Components

Assigning a unique identification (ID) to each person with access ensures that each individual is uniquely accountable for their actions. When such accountability is in place, actions taken on critical data and systems are performed by, and can be traced to, known and authorized users and processes.

Key sub-requirements:
- **8.1** Processes and mechanisms for identifying users and authenticating access to system components are defined and understood.
- **8.2** User identification and related accounts for users and administrators are strictly managed throughout the account lifecycle.
- **8.3** Strong authentication for users and administrators is established and managed. Passwords must be at least 12 characters (or 8 if the system does not support 12) containing both numeric and alphabetic characters. Passwords must be changed at least once every 90 days.
- **8.4** Multi-factor authentication (MFA) is implemented to secure access into the CDE for all non-console administrative access and all remote network access.
- **8.5** Multi-factor authentication (MFA) systems are configured to prevent misuse, including replay attacks.
- **8.6** Use of application and system accounts and associated authentication factors is strictly managed.

## Requirement 9: Restrict Physical Access to Cardholder Data

Any physical access to cardholder data or systems that store, process, or transmit cardholder data provides the opportunity for individuals to access and/or remove devices, data, systems or hardcopies, and should be appropriately restricted.

Key sub-requirements:
- **9.1** Processes and mechanisms for restricting physical access to cardholder data are defined and understood.
- **9.2** Physical access controls manage entry into facilities and systems containing cardholder data, using badge readers, locks, and video cameras.
- **9.3** Physical access for personnel and visitors is authorized and managed.
- **9.4** Media with cardholder data is securely stored, accessed, distributed, and destroyed when no longer needed.
- **9.5** Point-of-interaction (POI) devices are protected from tampering and unauthorized substitution through regular inspection and training of personnel.

## Requirement 10: Log and Monitor All Access to System Components and Cardholder Data

Logging mechanisms and the ability to track user activities are critical for preventing, detecting, or minimizing the impact of a data compromise. The presence of logs across all environments allows thorough tracking, alerting, and analysis when something does go wrong.

Key sub-requirements:
- **10.1** Processes and mechanisms for logging and monitoring all access to system components and cardholder data are defined and understood.
- **10.2** Audit logs are implemented to support the detection of anomalies and suspicious activity. Logs must record all individual user accesses to cardholder data, all actions taken by any individual with root or administrative privileges, access to all audit trails, invalid logical access attempts, and use of identification and authentication mechanisms.
- **10.3** Audit logs are protected from destruction and unauthorized modifications.
- **10.4** Audit logs are reviewed to identify anomalies or suspicious activity. Automated tools (e.g., SIEM) must be used to perform log reviews at least daily.
- **10.5** Audit log history is retained and available for analysis for at least one year, with a minimum of three months immediately available for analysis.
- **10.6** Time-synchronization technology is used to synchronize all critical system clocks using NTP or similar protocols.
- **10.7** Failures of critical security control systems are detected, reported, and responded to promptly.

## Requirement 11: Test Security of Systems and Networks Regularly

Vulnerabilities are being discovered continually by malicious individuals and researchers, and new software updates can introduce new vulnerabilities. System components, processes, and bespoke and custom software must be tested frequently to ensure security controls continue to reflect a changing environment.

Key sub-requirements:
- **11.1** Processes and mechanisms for regularly testing security of systems and networks are defined and understood.
- **11.2** Wireless access points are identified and monitored, and unauthorized wireless access points are addressed on a quarterly basis.
- **11.3** External and internal vulnerabilities are regularly identified, prioritized, and addressed. Internal vulnerability scans must be performed at least quarterly. External vulnerability scans must be performed at least quarterly by an Approved Scanning Vendor (ASV).
- **11.4** External and internal penetration testing is regularly performed at least annually and after any significant infrastructure or application change.
- **11.5** Network intrusions and unexpected file changes are detected and responded to using intrusion detection and/or prevention systems (IDS/IPS) and change-detection mechanisms.
- **11.6** Unauthorized changes on payment pages are detected and responded to using mechanisms that alert personnel to unauthorized modification.

## Requirement 12: Support Information Security with Organizational Policies and Programs

A strong security policy sets the security tone for the whole entity and informs personnel what is expected of them. All personnel should be aware of the sensitivity of cardholder data and their responsibilities for protecting it.

Key sub-requirements:
- **12.1** A comprehensive information security policy that governs and provides direction for protection of the entity's information assets is known and in effect.
- **12.2** Acceptable use policies for end-user technologies are defined and implemented.
- **12.3** Risks to the cardholder data environment are formally identified, evaluated, and managed through a formal risk assessment process performed at least annually.
- **12.4** PCI DSS compliance is managed through assignment of responsibility, documentation of scope, and annual validation.
- **12.5** PCI DSS scope is documented and validated. The entity must confirm the accuracy of scope at least annually.
- **12.6** Security awareness education is an ongoing activity for all personnel, delivered upon hire and at least once every 12 months.
- **12.7** Personnel are screened to reduce risks from insider threats through background checks prior to hire.
- **12.8** Risk to information assets from third-party service provider (TPSP) relationships is managed through written agreements, due diligence, and monitoring of compliance status.
- **12.9** Third-party service providers (TPSPs) support their customers' PCI DSS compliance through documented responsibilities and annual validation.
- **12.10** Suspected and confirmed security incidents that could impact the CDE are responded to immediately through a documented incident response plan that is tested at least annually.
