# PCI DSS v4.0 Compliance Implementation Guidelines

## Scope Determination

Before implementing PCI DSS controls, an organization must accurately determine the scope of its cardholder data environment (CDE). The CDE is comprised of people, processes, and technologies that store, process, or transmit cardholder data or sensitive authentication data, including any connected-to or security-impacting components.

### Defining the CDE Boundary

The CDE includes all system components that:
- Directly handle cardholder data (store, process, or transmit)
- Are in the same network segment as systems that handle cardholder data
- Are connected to or provide services to the CDE
- Could impact the security of the CDE

Network segmentation (also called network isolation) is not a PCI DSS requirement, but it is strongly recommended as a method to reduce the scope of the PCI DSS assessment and the cost of implementing and maintaining PCI DSS controls.

### Scope Validation

Organizations must validate their PCI DSS scope at least annually and upon significant changes to the environment. This validation should include:
- Identifying all locations and flows of cardholder data
- Identifying all system components in or connected to the CDE
- Confirming that all identified systems are included in the scope
- Documenting and confirming that any out-of-scope systems are properly segmented

## Data Protection Guidelines

### Cardholder Data Storage Rules

The following data elements have specific storage requirements under PCI DSS:

| Data Element | Storage Permitted | Protection Required | Req. Reference |
|---|---|---|---|
| Primary Account Number (PAN) | Yes | Must be rendered unreadable | 3.5 |
| Cardholder Name | Yes | Protect per PCI DSS requirements | 3.1-3.4 |
| Service Code | Yes | Protect per PCI DSS requirements | 3.1-3.4 |
| Expiration Date | Yes | Protect per PCI DSS requirements | 3.1-3.4 |
| Full Track Data | No | Cannot store after authorization | 3.3 |
| CAV2/CVC2/CVV2/CID | No | Cannot store after authorization | 3.3 |
| PIN/PIN Block | No | Cannot store after authorization | 3.3 |

### Acceptable Methods for Rendering PAN Unreadable

- **Strong cryptography** with associated key-management processes and procedures (e.g., AES-256)
- **One-way hash functions** based on strong cryptography (e.g., SHA-256) of the entire PAN
- **Truncation** (e.g., storing only the first six and last four digits)
- **Index tokens and pads** where the pads are securely stored

Important: The PAN must be rendered unreadable anywhere it is stored, including data on portable digital media, backup media, and in logs.

## Encryption Key Management

### Key Lifecycle Requirements

Organizations must implement comprehensive key management procedures that address the full lifecycle of cryptographic keys:

1. **Key Generation**: Keys must be generated using strong random number generators, with adequate key lengths (minimum 128-bit for symmetric encryption).

2. **Key Distribution**: Cryptographic keys must be distributed securely, never in clear-text form. Split knowledge and dual control procedures must be used for manual key management.

3. **Key Storage**: Keys must be stored in the fewest possible locations and in the most secure form, preferably in a Hardware Security Module (HSM) or similar tamper-resistant device.

4. **Key Rotation**: Cryptographic keys must be changed (rotated) at the end of their defined cryptoperiod, typically annually unless a longer period is justified by risk analysis.

5. **Key Revocation**: Procedures must exist for revoking keys when integrity has been weakened or keys are suspected of being compromised.

6. **Key Destruction**: Old or retired keys must be securely destroyed or rendered irrecoverable.

7. **Key Custodians**: Key custodians must formally acknowledge their key-custodian responsibilities through a signed form.

### Hardware Security Modules (HSMs)

HSMs are recommended for securing cryptographic operations and key storage. When using HSMs:
- HSMs must be physically secured and access-controlled
- Firmware and configuration must be kept current
- Audit logging of all HSM operations must be enabled
- Backup and recovery procedures must be documented and tested

## Access Control Implementation

### Role-Based Access Control (RBAC)

PCI DSS requires implementing role-based access control with these principles:

- **Least Privilege**: Users must only have access to the minimum resources necessary for their job function.
- **Need to Know**: Access to cardholder data must be restricted to individuals whose job requires such access.
- **Default Deny**: Access control systems must be set to "deny all" unless explicitly allowed.
- **Separation of Duties**: No single individual should control all aspects of a critical transaction or process.

### Authentication Requirements (v4.0 Updates)

PCI DSS v4.0 strengthened authentication requirements:

- **Password Length**: Minimum 12 characters (increased from 7 in v3.2.1), or 8 characters if the system does not support 12.
- **Password Complexity**: Must contain both numeric and alphabetic characters.
- **Password History**: Must not match any of the last four passwords used.
- **Account Lockout**: After no more than 10 invalid login attempts, the account must be locked for a minimum of 30 minutes or until an administrator enables the user ID.
- **Session Timeout**: Idle sessions must time out after no more than 15 minutes.
- **MFA**: Required for all access into the CDE, not just remote access (expanded from v3.2.1).

## Vulnerability Management

### Scanning Requirements

- **Internal Vulnerability Scans**: Must be performed at least quarterly and after any significant change. High-risk and critical vulnerabilities (per CVSS) must be resolved, with rescans to confirm.
- **External Vulnerability Scans**: Must be performed at least quarterly by a PCI SSC Approved Scanning Vendor (ASV). Four passing scans within 12 months are required.
- **Authenticated Internal Scans**: New in v4.0, internal scans must include authenticated scanning sufficient to detect vulnerabilities that require authentication.

### Penetration Testing

- External penetration testing must be performed at least annually and after significant changes.
- Internal penetration testing must be performed at least annually and after significant changes.
- Testing must cover the entire CDE perimeter and critical systems.
- Both network-layer and application-layer testing must be performed.
- Tests must validate that segmentation controls are operational and effective (if used to reduce scope).
- Findings must be corrected and retested to verify the corrections.

## Incident Response

### Incident Response Plan Requirements

PCI DSS Requirement 12.10 mandates that organizations maintain a documented incident response plan that includes:

1. Roles, responsibilities, and communication strategies for breach notification
2. Specific incident response procedures for different types of incidents
3. Business recovery and continuity procedures
4. Data backup processes
5. Analysis of legal requirements for reporting compromises
6. Coverage and responses for all critical system components
7. Reference or inclusion of incident response procedures from payment brands

The incident response plan must be tested at least annually through tabletop exercises or simulated incidents. Staff with security breach responsibilities must be trained at least annually on their incident response duties.

## Compliance Validation

### Self-Assessment Questionnaire (SAQ)

Organizations that are eligible for self-assessment (typically smaller merchants) must complete the appropriate SAQ annually:

- **SAQ A**: For e-commerce or mail/telephone-order merchants with all cardholder data functions outsourced
- **SAQ A-EP**: For e-commerce merchants with a website that doesn't directly receive cardholder data but impacts the security of the payment transaction
- **SAQ B**: For merchants using only imprint machines or standalone dial-out terminals
- **SAQ C**: For merchants with payment application systems connected to the Internet
- **SAQ D**: For all other merchants and for all service providers

### Report on Compliance (ROC)

Level 1 merchants and service providers must undergo an annual on-site assessment by a Qualified Security Assessor (QSA) and submit a Report on Compliance (ROC) to their acquiring bank or payment brand.

## Customized Approach (New in v4.0)

PCI DSS v4.0 introduces the customized approach as an alternative to the defined approach for meeting requirements. Under the customized approach:

- Organizations define their own controls to meet the stated security objective of each requirement
- A targeted risk analysis must document why the customized control is adequate
- The QSA must validate the effectiveness of the customized control
- This approach provides flexibility but requires more rigorous documentation and testing
- Not all requirements are eligible for the customized approach
