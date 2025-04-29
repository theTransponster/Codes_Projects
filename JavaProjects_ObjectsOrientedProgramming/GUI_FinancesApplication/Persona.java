public class Persona {
	public double sueldoMensual,
	               aguinaldo,
	               primaVacacional,
	               colegiatura,
	               SGM,
	               funerarios,
	               medHosp,
	               hipoteca,
	               donativos,
	               subRetiro,
	               Transporte,
	               dEducacion;
	               
	public double primaExenta=1209.00;
	public String Nombre,
				   RFC,
				   educacion;
	
	public Persona(String Nombre, String RFC, double sueldoMensual, 
				   double aguinaldo, double primaVaca, String educacion,
				   double Colegiatura, double SGM, double funerarios, double medHosp,
				   double hipoteca,double donativos, double subRetiro, double Transporte){
		this.Nombre=Nombre;
		this.RFC=RFC;
		this.sueldoMensual=sueldoMensual;
		this.aguinaldo=aguinaldo;
		this.primaVacacional=primaVaca;
		this.educacion=educacion;
		this.colegiatura=Colegiatura;
		this.SGM=SGM;
		this.funerarios=funerarios;
		this.medHosp=medHosp;
		this.hipoteca=hipoteca;
		this.donativos=donativos;
		this.subRetiro=subRetiro;
		this.Transporte=Transporte; 
	}
	public double ingresoAnual(){
		return this.sueldoMensual*12;
	}
	public double aguinaldoExento(){
		return this.sueldoMensual/2;
	}
	public double aguinaldoGravado(){
		return this.aguinaldo-this.aguinaldoExento();
	}
	
	public double primaGravado(){
		return this.primaVacacional-this.primaExenta;
	}
	public double getTIngresosGravan(){
		return this.ingresoAnual()+this.aguinaldoGravado()+this.primaGravado();
	}
	public double getTotatlIngresos(){
		return this.ingresoAnual()+this.aguinaldo+this.primaVacacional;
	}
	public double deduccionColegiatura(){
		if(this.educacion.equals("Primaria")){
			this.dEducacion=12900.00;
		}else if(this.educacion.equals("Preescolar")){
			this.dEducacion=14200.00;
		}else if(this.educacion.equals("Secundaria")){
			this.dEducacion=19900;
		}else if(this.educacion.equals("Preparatoria")){
			this.dEducacion=17100.00;
		}else if(this.educacion.equals("Superior")){
			this.dEducacion=24500.00;
		}else if(this.educacion.equals("Ninguna")){
			this.dEducacion=0.00;
		}
		return this.dEducacion;	
	}
	public double getTotalDeducciones(){
		if(this.colegiatura>this.dEducacion){
			this.colegiatura=this.dEducacion;
		}
		return this.SGM+this.funerarios+this.medHosp+this.hipoteca+this.donativos+this.Transporte+this.colegiatura;
	}
	
}
