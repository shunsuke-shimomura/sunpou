//! Cross-validation with uom: verify that sunpou's dimension arithmetic
//! produces the same numerical results as uom for representative calculations.

use sunpou::aliases::*;
use sunpou::scalar::Scalar;

#[test]
fn velocity_from_length_div_time() {
    use uom::si::f64::{Length as UomLength, Time as UomTime, Velocity as UomVelocity};
    use uom::si::length::meter;
    use uom::si::time::second;
    use uom::si::velocity::meter_per_second;

    let uom_len = UomLength::new::<meter>(100.0);
    let uom_time = UomTime::new::<second>(5.0);
    let uom_vel: UomVelocity = uom_len / uom_time;

    let our_len = Scalar::<Length>::from_raw(100.0);
    let our_time = Scalar::<Time>::from_raw(5.0);
    let our_vel: Scalar<Velocity> = our_len / our_time;

    assert_eq!(uom_vel.get::<meter_per_second>(), our_vel.into_raw());
}

#[test]
fn force_from_mass_times_accel() {
    use uom::si::acceleration::meter_per_second_squared;
    use uom::si::f64::{
        Acceleration as UomAccel, Force as UomForce, Mass as UomMass,
    };
    use uom::si::force::newton;
    use uom::si::mass::kilogram;

    let uom_mass = UomMass::new::<kilogram>(10.0);
    let uom_accel = UomAccel::new::<meter_per_second_squared>(9.8);
    let uom_force: UomForce = uom_mass * uom_accel;

    let our_mass = Scalar::<Mass>::from_raw(10.0);
    let our_accel = Scalar::<Acceleration>::from_raw(9.8);
    let our_force: Scalar<Force> = our_mass * our_accel;

    assert_eq!(uom_force.get::<newton>(), our_force.into_raw());
}

#[test]
fn energy_from_force_times_length() {
    use uom::si::energy::joule;
    use uom::si::f64::{Energy as UomEnergy, Force as UomForce, Length as UomLength};
    use uom::si::force::newton;
    use uom::si::length::meter;

    let uom_force = UomForce::new::<newton>(50.0);
    let uom_len = UomLength::new::<meter>(3.0);
    let uom_energy: UomEnergy = uom_force * uom_len;

    let our_force = Scalar::<Force>::from_raw(50.0);
    let our_len = Scalar::<Length>::from_raw(3.0);
    let our_energy: Scalar<Energy> = our_force * our_len;

    assert_eq!(uom_energy.get::<joule>(), our_energy.into_raw());
}

#[test]
fn momentum_from_mass_times_velocity() {
    use uom::si::f64::{Mass as UomMass, Momentum as UomMomentum, Velocity as UomVelocity};
    use uom::si::mass::kilogram;
    use uom::si::momentum::kilogram_meter_per_second;
    use uom::si::velocity::meter_per_second;

    let uom_mass = UomMass::new::<kilogram>(5.0);
    let uom_vel = UomVelocity::new::<meter_per_second>(20.0);
    let uom_mom: UomMomentum = uom_mass * uom_vel;

    let our_mass = Scalar::<Mass>::from_raw(5.0);
    let our_vel = Scalar::<Velocity>::from_raw(20.0);
    let our_mom: Scalar<Momentum> = our_mass * our_vel;

    assert_eq!(
        uom_mom.get::<kilogram_meter_per_second>(),
        our_mom.into_raw()
    );
}

#[test]
fn acceleration_from_velocity_div_time() {
    use uom::si::acceleration::meter_per_second_squared;
    use uom::si::f64::{
        Acceleration as UomAccel, Time as UomTime, Velocity as UomVelocity,
    };
    use uom::si::time::second;
    use uom::si::velocity::meter_per_second;

    let uom_vel = UomVelocity::new::<meter_per_second>(30.0);
    let uom_time = UomTime::new::<second>(3.0);
    let uom_accel: UomAccel = uom_vel / uom_time;

    let our_vel = Scalar::<Velocity>::from_raw(30.0);
    let our_time = Scalar::<Time>::from_raw(3.0);
    let our_accel: Scalar<Acceleration> = our_vel / our_time;

    assert_eq!(
        uom_accel.get::<meter_per_second_squared>(),
        our_accel.into_raw()
    );
}

#[test]
fn power_from_energy_div_time() {
    use uom::si::energy::joule;
    use uom::si::f64::{Energy as UomEnergy, Power as UomPower, Time as UomTime};
    use uom::si::power::watt;
    use uom::si::time::second;

    let uom_energy = UomEnergy::new::<joule>(1000.0);
    let uom_time = UomTime::new::<second>(10.0);
    let uom_power: UomPower = uom_energy / uom_time;

    let our_energy = Scalar::<Energy>::from_raw(1000.0);
    let our_time = Scalar::<Time>::from_raw(10.0);
    let our_power: Scalar<Power> = our_energy / our_time;

    assert_eq!(uom_power.get::<watt>(), our_power.into_raw());
}

#[test]
fn torque_from_force_times_length() {
    use uom::si::f64::{Force as UomForce, Length as UomLength, Torque as UomTorque};
    use uom::si::force::newton;
    use uom::si::length::meter;
    use uom::si::torque::newton_meter;

    let uom_force = UomForce::new::<newton>(25.0);
    let uom_len = UomLength::new::<meter>(2.0);
    let uom_torque: UomTorque = (uom_force * uom_len).into();

    let our_force = Scalar::<Force>::from_raw(25.0);
    let our_len = Scalar::<Length>::from_raw(2.0);
    let our_torque: Scalar<Torque> = our_force * our_len;

    assert_eq!(uom_torque.get::<newton_meter>(), our_torque.into_raw());
}

#[test]
fn chained_operations() {
    // v = d / t, F = m * a, W = F * d, P = W / t
    use uom::si::f64::{
        Acceleration as UomAccel, Length as UomLength, Mass as UomMass,
        Power as UomPower, Time as UomTime,
    };
    use uom::si::acceleration::meter_per_second_squared;
    use uom::si::length::meter;
    use uom::si::mass::kilogram;
    use uom::si::power::watt;
    use uom::si::time::second;

    let m = 10.0;
    let a = 5.0;
    let d = 20.0;
    let t = 4.0;

    // uom chain
    let uom_force = UomMass::new::<kilogram>(m) * UomAccel::new::<meter_per_second_squared>(a);
    let uom_work = uom_force * UomLength::new::<meter>(d);
    let uom_power: UomPower = uom_work / UomTime::new::<second>(t);

    // our chain
    let our_force: Scalar<Force> =
        Scalar::<Mass>::from_raw(m) * Scalar::<Acceleration>::from_raw(a);
    let our_work: Scalar<Energy> = our_force * Scalar::<Length>::from_raw(d);
    let our_power: Scalar<Power> = our_work / Scalar::<Time>::from_raw(t);

    assert_eq!(uom_power.get::<watt>(), our_power.into_raw());
}
