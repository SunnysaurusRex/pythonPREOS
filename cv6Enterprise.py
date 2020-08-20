import numpy as np
import cmath as m
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import newton

R = 8.314462; #Gas constant, MPa cm**3 / mol-K
##constants and parameters
T = 298; # temperature of interest, K
P = .1; #pressure of interest, MPa

Tc = 369.8; #critical temperature
Pc = 4.249; #critical pressure
w =  0.152; #acentric factor
Tt = 85; #triple point temperature 
Pt = 1.685e-10; #triple point pressure
#peng-robinson parameters
Tr = T/Tc; #reduced temperature
ac = 0.45724*R**2*Tc**2/Pc; #a, critical
k = 0.37464 + 1.54226*w - 0.269932*w**2; #kappa 
al = (1 + k*(1-Tr**.5))**2; #a, function of T 
a = ac*al; #alpha
b = 0.07780*R*Tc/Pc; #b, theoretical volume
#antoine constants, pressure = exp(A + B/T + C*lnT + D*T^E)
c1 = 59.078; #A
c2 = -3492.6; #B
c3 = -6.0669; #C
c4 = 1.0919E-05; #D
c5 = 2; #E

##functions
def freikugel(P): #calculates molar volume at a given pressure (MPa)
	#standard form of a cubic equation, rx^3 + sx^2 + tx + u = 0
	r = -P;
	s = R*T - P*b;
	t = 2*b*R*T - a + 3*b**2*P;
	u = a*b - P*b**3 - R*T*b**2;
	#depressed cubic y^3 + py + q = 0, substitution (x = y - s/3r) 
	p = (t - s**2/3/r)/r;
	q = (u + 2*s**3/27/r**2 - s*t/3/r)/r;
	radicand = q**2 + 4/27*p**3;
	#print("rad = ", radicand)
	if radicand<0:
		W = (-q + np.sqrt(np.absolute(radicand))*1j)/2;
	else:
		W = (-q + np.sqrt(radicand))/2;
	theta = m.phase(W);
	v = abs(W)**(1/3);
	solutions = [];
	for i in range(0,3):
		z = v*(np.cos((theta + 2*np.pi*i)/3) + np.sin((theta + 2*np.pi*i)/3)*1j);
		y = z - p/3/z;
		root = y - s/3/r;
		solutions.append(root.real);
	Vv = max(solutions); #molar volume of vapor 
	Vl = min(solutions); #molar volume of liquid
	return [Vl, Vv];

def fugac(P, V): #returns fugacity of a liquid/vapor at a given presure and a known molar volume
	A = a*P/R**2/T**2;
	B = b*P/R/T;
	Z = V/(V-b) -a*V/R/T/( V*(V+b) + b*(V-b));
	f = P*np.exp((Z-1) - np.log(Z-B) - A/B/2**1.5*np.log( (Z+(1+np.sqrt(2))*B) / (Z+(1-np.sqrt(2))*B)) );
	return f

#interpolate a possible vapor pressure
xp = [Tt, Tc]
yp = [np.log10(Pt), np.log10(Pc)];
guess = np.power(10, np.interp(T, xp, yp));

#combination of freikugel and fugac; the function to solve for 
def optim(P):
	r = -P;
	s = R*T - P*b;
	t = 2*b*R*T - a + 3*b**2*P;
	u = a*b - P*b**3 - R*T*b**2;
	p = (t - s**2/3/r)/r;
	q = (u + 2*s**3/27/r**2 - s*t/3/r)/r;
	radicand = q**2 + 4/27*p**3;
	#print("rad = ", radicand)
	if radicand<0:
		W = (-q + np.sqrt(np.absolute(radicand))*1j)/2;
	else:
		W = (-q + np.sqrt(radicand))/2;
	theta = m.phase(W);
	v = abs(W)**(1/3);
	solutions = [];
	for i in range(0,3):
		z = v*(np.cos((theta + 2*np.pi*i)/3) + np.sin((theta + 2*np.pi*i)/3)*1j);
		y = z - p/3/z;
		root = y - s/3/r;
		solutions.append(root.real);
	Vv = max(solutions);
	Vl = min(solutions);
	A = a*P/R**2/T**2;
	B = b*P/R/T;
	Zl = Vl/(Vl-b) -a*Vl/R/T/( Vl*(Vl+b) + b*(Vl-b));
	Zv = Vv/(Vv-b) -a*Vv/R/T/( Vv*(Vv+b) + b*(Vv-b));
	fl = P*np.exp((Zl-1) - np.log(Zl-B) - A/B/2**1.5*np.log( (Zl+(1+np.sqrt(2))*B) / (Zl+(1-np.sqrt(2))*B)) );
	fv = P*np.exp((Zv-1) - np.log(Zv-B) - A/B/2**1.5*np.log( (Zv+(1+np.sqrt(2))*B) / (Zv+(1-np.sqrt(2))*B)) );
	return fl-fv

#comparing result to a more accurate value that is based on data
def antoine(T): #returns vapor pressure(MPa) calculated by the Antoine equation 
	vaporpressure = np.exp(c1 + c2/T +c3*np.log(T) + c4*T**c5); #in Pa
	return vaporpressure*1e-6 #convert to MPa

def percentError(x, accepted):
	return np.absolute(x-accepted)/accepted*100
##plotting 
def plotCubic():
	plt.figure(1)
	plt.title("Peng-Robinson Cubic Equation of State", loc ="left")
	plt.ylabel("Pressure (MPa)")
	plt.xlabel("Molar Volume (cm$^{3}$/mol)")
	plt.axhline()
	plt.ylim(0, Pc)
	plt.axvline(b, label="b = " + str(round(b,3)));
	V = np.linspace(b+1, 2000, 200)
	P = R*T/(V-b) - a/(V*(V+b) + b*(V-b));
	line = plt.plot(V, P, color="salmon", label="T = " + str(T) + " K");
	plt.legend()	

def plotFugacityDifference():
	P = np.linspace(Pt, Pc, 100, endpoint=False);
	liquid, vapor, pressure, fugadiff = [],[],[],[];
	for i in range(0, len(P)):
		volumes = freikugel(P[i]);
		try:
			fl = fugac(P[i], volumes[0]);
			fv = fugac(P[i], volumes[1]);
			liquid.append(volumes[0]);
			vapor.append(volumes[1]);
			pressure.append(P[i]);
			fugadiff.append(fl-fv);
		except:
			pass
	plt.figure(2);
	plt.title('Fugacity Difference: $f_{l}-f_{v}$ vs Pressure at ' + str(T) +' K');
	plt.ylabel('$f_{l}-f_{v}$ (MPa)');
	plt.xlabel('Pressure (MPa)');
	line2 = plt.plot(pressure, fugadiff, color='magenta', label='$f_{l}-f_{v}$');
	plt.legend();
	plt.ylim(-2,2)
	plt.axhline(color="blue")	
##function call
Pvap = newton(optim, guess, maxiter=50);
volumes = freikugel(P);
fugacityLiquid = volumes[0];
fugactityVapor = volumes[1];
print("\nT = ", T);
print("	liquid molar volume: %f cm^3/mol" %(volumes[0]) );
print("	vapor molar volume: %f cm^3/mol" %(volumes[1]) );
print("	liquid fugacity: %f " %(fugac(P, volumes[0])));
print("	vapor fugacity: %f" %(fugac(P, volumes[1])));
print("\n")
plotCubic();
plotFugacityDifference();
print("\n", "At %.2f K, VLE exists at %f MPa" %(T, Pvap))
plt.show();
